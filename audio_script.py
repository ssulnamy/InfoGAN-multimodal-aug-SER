"""Stage 2: train the audio-text model on top of a frozen text encoder.

The Stage-1 text encoder (``text_script.py``) is loaded frozen. The audio side is
trained in three sub-stages, selected with ``--stage``:

    align    (train1)  align the audio transformer to text (CE + two InfoNCE losses)
    infogan  (train2)  train the InfoGAN feature generator (D / G / Q steps)
    fusion   (train3)  train the audio-text fusion classifier with augmentation

Each sub-stage loads the audio checkpoint produced by the previous one and writes a
new one, so the intended order is: align -> infogan -> fusion.

Examples
--------
    python audio_script.py --stage align   --text-ckpt checkpoints/text.pt
    python audio_script.py --stage infogan --text-ckpt checkpoints/text.pt
    python audio_script.py --stage fusion  --text-ckpt checkpoints/text.pt
"""

import argparse
import os
import warnings
from datetime import datetime

import numpy as np
import torch
from accelerate import Accelerator
from info_nce import InfoNCE
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import Wav2Vec2Config, Wav2Vec2Processor

import common
from measure_WA_UA import unweighted_accuracy, uar_score, weighted_accuracy
from src.aug_bal import aug_bal
from src.datacollator import DataCollatorCTCWithPadding
from src.dataset import Dataset_aug

warnings.filterwarnings("ignore")

GRAD_ACCUM = 2
EARLY_STOP_PATIENCE = 100

# Per-stage default checkpoint flow: each stage reads the previous stage's output.
STAGE_DEFAULTS = {
    "align":   {"audio_in": None,                          "audio_out": "checkpoints/audio_align.pt"},
    "infogan": {"audio_in": "checkpoints/audio_align.pt",  "audio_out": "checkpoints/audio_infogan.pt"},
    "fusion":  {"audio_in": "checkpoints/audio_infogan.pt", "audio_out": "checkpoints/audio_fusion.pt"},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2: train the audio-text model (align / infogan / fusion).")
    common.add_common_args(parser)
    parser.add_argument("--stage", required=True, choices=["align", "infogan", "fusion"])
    parser.add_argument(
        "--text-ckpt", default="checkpoints/text.pt",
        help="Frozen Stage-1 text encoder (required; run text_script.py first).")
    parser.add_argument(
        "--audio-ckpt-in", default=None,
        help="Audio checkpoint to resume from (default: previous stage's output).")
    parser.add_argument(
        "--audio-ckpt-out", default=None,
        help="Where to save this stage's audio model (default: per-stage path).")
    args = parser.parse_args()

    if args.audio_ckpt_in is None:
        args.audio_ckpt_in = STAGE_DEFAULTS[args.stage]["audio_in"]
    if args.audio_ckpt_out is None:
        args.audio_ckpt_out = STAGE_DEFAULTS[args.stage]["audio_out"]
    if args.epochs is None:
        args.epochs = 10
    if args.batch_size is None:
        args.batch_size = 8
    if args.data_root is None:
        args.data_root = common.DEFAULT_DATA_ROOT[args.dataset]

    if not os.path.exists(args.text_ckpt):
        parser.error(
            f"--text-ckpt '{args.text_ckpt}' not found; run Stage 1 (text_script.py) first.")
    if args.stage in ("infogan", "fusion") and not (
            args.audio_ckpt_in and os.path.exists(args.audio_ckpt_in)):
        parser.error(
            f"--audio-ckpt-in '{args.audio_ckpt_in}' not found; run the previous stage first.")
    return args


def make_loader(items, tokenizer, collator, batch_size, shuffle, drop_last=False):
    """Wrap raw examples in an audio dataset and DataLoader."""
    dataset = Dataset_aug(list(items))
    dataset.set_tokenizer(tokenizer)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator,
        num_workers=0, drop_last=drop_last, pin_memory=False)


def main():
    args = parse_args()
    torch.set_num_threads(1)

    accelerator = Accelerator(
        device_placement=False, mixed_precision="fp16",
        gradient_accumulation_steps=GRAD_ACCUM)
    device = accelerator.device

    dataset = common.build_dataset(args.dataset, args.data_root, args.refresh_cache)
    num_labels = dataset.num_labels
    train_examples, val_examples, test_examples = common.resolve_splits(dataset, args.fold, lim=args.lim)

    # Audio backbone config / processor.
    config_a = Wav2Vec2Config.from_pretrained(args.audio_backbone, num_labels=num_labels)
    config_a.proj_size = config_a.hidden_size
    processor = Wav2Vec2Processor.from_pretrained(args.audio_backbone)

    # Class weights (shared by the CE loss and the InfoGAN discrete loss).
    weight = common.class_weights(args.dataset, num_labels, device)
    reg_fct = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=0.0)
    info_nce = InfoNCE()

    # Frozen Stage-1 text encoder.
    txt, config_t, tokenizer = common.build_text_encoder(
        args.text_backbone, num_labels, device, freeze=True, ckpt=args.text_ckpt)

    # Audio model (fresh from backbone for `align`, resumed for later stages).
    model = common.build_audio_encoder(
        config_a, num_labels, args.audio_backbone, ckpt=args.audio_ckpt_in, ce_weight=weight)
    model.audio_wav2vec2.wav2vec2.freeze_feature_encoder()
    model.audio_wav2vec2.gradient_checkpointing_enable()
    model.to(device)

    collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    training_loader = make_loader(
        train_examples, tokenizer, collator, args.batch_size, shuffle=True, drop_last=True)
    val_loader = make_loader(
        val_examples, tokenizer, collator, args.batch_size, shuffle=True)
    testing_loader = make_loader(
        test_examples, tokenizer, collator, args.batch_size, shuffle=True)

    # Stage-specific optimizers over the model's named parameter groups.
    p = model.par_dic
    optimizers = {}
    if args.stage == "align":
        optimizers["opt1"] = torch.optim.Adam(params=p["D"] + p["T"], lr=1e-4)
    elif args.stage == "infogan":
        optimizers["optD"] = torch.optim.Adam(params=p["D"], lr=1e-4, betas=(0.5, 0.999))
        optimizers["optG"] = torch.optim.Adam(params=p["G"], lr=1e-4, betas=(0.5, 0.999))
        optimizers["optQ"] = torch.optim.Adam(params=p["G"] + p["D"], lr=1e-4, betas=(0.5, 0.999))
    else:  # fusion
        optimizers["opt2"] = torch.optim.Adam(params=p["C"] + p["D"], lr=1e-4)

    prepared = accelerator.prepare(
        model, txt, training_loader, val_loader, testing_loader, *optimizers.values())
    model, txt, training_loader, val_loader, testing_loader = prepared[:5]
    for name, opt in zip(optimizers, prepared[5:]):
        optimizers[name] = opt

    logger = SummaryWriter(
        os.path.join(args.log_dir, f"stage_{args.stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))

    def eval_loss():
        """Validation CE on the audio-only logits (used by align / infogan)."""
        model.eval()
        total = 0.0
        with torch.no_grad():
            with tqdm(enumerate(val_loader), unit="batch",
                      total=len(val_loader), dynamic_ncols=True) as tepoch:
                for step, data in tepoch:
                    tepoch.set_description("Val")
                    labels = data["labels"].to(device).view(-1, model.num_labels)
                    outputs = model(data, device)
                    total += reg_fct(outputs.logits.view(-1, model.num_labels), labels).item()
                    tepoch.set_postfix(loss=total / (step + 1))
        return total / len(val_loader)

    def train_align(epoch):
        model.train()
        total_loss = 0.0
        with tqdm(enumerate(training_loader), unit="batch",
                  total=len(training_loader), dynamic_ncols=True) as tepoch:
            for step, data in tepoch:
                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        tepoch.set_description(f"Epoch {epoch}")
                        labels = data["labels"].to(device).view(-1, model.num_labels)
                        t, _ = txt(data, device)
                        outputs = model(data, device, t=t)

                        step_idx = step + epoch * len(training_loader)
                        loss = reg_fct(outputs.logits.view(-1, model.num_labels), labels)
                        logger.add_scalar("train/E", loss.item(), step_idx)
                        closs = info_nce(t, outputs.rep_real)
                        logger.add_scalar("train/CL", closs.item(), step_idx)
                        tloss = info_nce(t, outputs.t_)
                        logger.add_scalar("train/TMI", tloss.item(), step_idx)
                        loss = loss + closs + tloss

                        optimizers["opt1"].zero_grad()
                        accelerator.backward(loss)
                        optimizers["opt1"].step()

                        total_loss += loss.item()
                        tepoch.set_postfix(loss=total_loss / (step + 1))
        return eval_loss()

    def train_infogan(epoch):
        model.train()
        total_loss = 0.0
        with tqdm(enumerate(training_loader), unit="batch",
                  total=len(training_loader), dynamic_ncols=True) as tepoch:
            for step, data in tepoch:
                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        tepoch.set_description(f"Epoch {epoch}")
                        step_idx = step + epoch * len(training_loader)

                        # Discriminator step.
                        t, _ = txt(data, device)
                        outputs = model(data, device, t=t)
                        loss = outputs.Dloss
                        optimizers["optD"].zero_grad()
                        accelerator.backward(loss)
                        optimizers["optD"].step()
                        logger.add_scalar("train/D", loss.item(), step_idx)
                        total_loss += loss.item()

                        # Generator step.
                        t, _ = txt(data, device)
                        outputs = model(data, device, t=t)
                        loss = outputs.Gloss
                        optimizers["optG"].zero_grad()
                        accelerator.backward(loss)
                        optimizers["optG"].step()
                        logger.add_scalar("train/G", loss.item(), step_idx)
                        total_loss += loss.item()

                        # Info (Q) step.
                        t, _ = txt(data, device)
                        outputs = model(data, device, t=t)
                        loss = outputs.Qloss
                        optimizers["optQ"].zero_grad()
                        accelerator.backward(loss[0])
                        optimizers["optQ"].step()
                        logger.add_scalar("train/Qd", loss[1].item(), step_idx)
                        logger.add_scalar("train/Qc", loss[2].item(), step_idx)
                        total_loss += loss[0].item()

                        tepoch.set_postfix(loss=total_loss / (step + 1))
                        logger.add_scalar("train/total", total_loss / (step + 1), step_idx)
        return eval_loss()

    def train_fusion(epoch):
        model.train()
        total_loss = 0.0
        with tqdm(enumerate(training_loader), unit="batch",
                  total=len(training_loader), dynamic_ncols=True) as tepoch:
            for step, data in tepoch:
                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        tepoch.set_description(f"Epoch {epoch}")
                        labels = data["labels"].to(device).view(-1, model.num_labels)
                        t, _ = txt(data, device)
                        outputs = model(data, device, t=t)

                        # CE over every audio / fused / generated head, plus the
                        # augmentation-balancing term.
                        loss = reg_fct(outputs.logits.view(-1, model.num_labels), labels)
                        loss += reg_fct(outputs.logits_fake.view(-1, model.num_labels), labels)
                        loss += reg_fct(outputs.logits_fuse.view(-1, model.num_labels), labels)
                        loss += reg_fct(outputs.logits_fusef.view(-1, model.num_labels), labels)
                        loss += reg_fct(outputs.logits_fuseft.view(-1, model.num_labels), labels)
                        loss += reg_fct(outputs.logits_fuset.view(-1, model.num_labels), labels)
                        loss += aug_bal(model, labels, device)

                        optimizers["opt2"].zero_grad()
                        accelerator.backward(loss)
                        optimizers["opt2"].step()

                        total_loss += loss.item()
                        tepoch.set_postfix(loss=total_loss / (step + 1))

    def evaluate():
        """Test-set predictions using the stage-appropriate output head."""
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            with tqdm(enumerate(testing_loader), unit="batch",
                      total=len(testing_loader), dynamic_ncols=True) as tepoch:
                for _, data in tepoch:
                    tepoch.set_description("Test")
                    if args.stage == "fusion":
                        t, _ = txt(data, device)
                        logits = model(data, device, t=t).logits_fuse
                    else:
                        logits = model(data, device).logits
                    preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
                    targets.extend(torch.argmax(data["labels"], dim=-1).cpu().numpy().tolist())
        return preds, targets

    def save_checkpoint():
        if not accelerator.is_main_process:
            return
        os.makedirs(os.path.dirname(args.audio_ckpt_out) or ".", exist_ok=True)
        torch.save(accelerator.unwrap_model(model).state_dict(), args.audio_ckpt_out)
        print(f"saved audio model to {args.audio_ckpt_out}")

    # ---- Training ----
    if args.stage in ("align", "infogan"):
        train_fn = train_align if args.stage == "align" else train_infogan
        min_loss = float("inf")
        patience = EARLY_STOP_PATIENCE
        for epoch in range(args.epochs):
            val_loss = train_fn(epoch)
            if val_loss < min_loss:
                min_loss = val_loss
                patience = EARLY_STOP_PATIENCE
            elif patience > 0:
                patience -= 1
            else:
                print(f"early stop (best val loss {min_loss:.4f})")
                break
    else:  # fusion: no validation head -> train the full schedule
        for epoch in range(args.epochs):
            train_fusion(epoch)

    save_checkpoint()

    # ---- Evaluation ----
    preds, targets = evaluate()
    preds, targets = np.array(preds), np.array(targets)
    print(f"WA  = {weighted_accuracy(list_y_true=targets, list_y_pred=preds)}")
    print(f"UA  = {unweighted_accuracy(list_y_true=targets, list_y_pred=preds)}")
    print(f"UAR = {uar_score(labels=targets, predictions=preds)}")

    logger.flush()
    logger.close()


if __name__ == "__main__":
    main()
