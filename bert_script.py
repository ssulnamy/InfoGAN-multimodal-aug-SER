"""Stage 1: train the text transformer that predicts emotion from text.

The trained encoder is saved to ``--text-ckpt`` and later loaded (frozen) by the
audio-text stages in ``aae_script.py``.

Example
-------
    python bert_script.py --dataset msp-podcast --text-ckpt checkpoints/text.pt
"""

import argparse
import os
import warnings
from datetime import datetime

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import common
from measure_WA_UA import unweighted_accuracy, uar_score, weighted_accuracy
from src.datacollator import TXTDataCollatorWithPadding
from src.dataset import TXTDataset_aug

warnings.filterwarnings("ignore")

GRAD_ACCUM = 2
EARLY_STOP_PATIENCE = 100


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1: train the text emotion classifier "
                    "(Llama + LoRA, BERT or RoBERTa backbone).")
    common.add_common_args(parser)
    parser.add_argument(
        "--text-ckpt", default="checkpoints/text.pt",
        help="Where to save the trained encoder (and to resume from if it exists).")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--do-train", action="store_true", help="Run training.")
    parser.add_argument("--do-eval", action="store_true", help="Run test-set evaluation.")
    args = parser.parse_args()

    # Default to doing both when neither flag is given.
    if not args.do_train and not args.do_eval:
        args.do_train = args.do_eval = True
    if args.epochs is None:
        args.epochs = 100
    if args.batch_size is None:
        args.batch_size = 128
    if args.data_root is None:
        args.data_root = common.DEFAULT_DATA_ROOT[args.dataset]
    return args


def make_loader(items, tokenizer, collator, batch_size, shuffle, drop_last=False):
    """Wrap raw examples in a text dataset and DataLoader."""
    dataset = TXTDataset_aug(list(items))
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

    txt, config_t, tokenizer = common.build_text_encoder(
        args.text_backbone, num_labels, device, freeze=False, ckpt=args.text_ckpt)

    collator = TXTDataCollatorWithPadding(padding=True)
    weight = common.class_weights(args.dataset, num_labels, device)
    reg_fct = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=0.0)

    training_loader = make_loader(
        train_examples, tokenizer, collator, args.batch_size, shuffle=True, drop_last=True)
    val_loader = make_loader(
        val_examples, tokenizer, collator, args.batch_size, shuffle=True)
    testing_loader = make_loader(
        test_examples, tokenizer, collator, args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(params=txt.parameters(), lr=args.lr)

    txt, training_loader, val_loader, testing_loader, optimizer = accelerator.prepare(
        txt, training_loader, val_loader, testing_loader, optimizer)

    logger = SummaryWriter(
        os.path.join(args.log_dir, f"stage1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))

    def train_one_epoch(epoch):
        txt.train()
        total_loss = 0.0
        with tqdm(enumerate(training_loader), unit="batch",
                  total=len(training_loader), dynamic_ncols=True) as tepoch:
            for step, data in tepoch:
                with accelerator.accumulate(txt):
                    with accelerator.autocast():
                        tepoch.set_description(f"Epoch {epoch}")
                        labels = data["labels"].to(device).view(-1, num_labels)
                        _, logits = txt(data, device)
                        loss = reg_fct(logits.view(-1, num_labels), labels)

                        logger.add_scalar("train/TXT", loss.item(),
                                          step + epoch * len(training_loader))
                        optimizer.zero_grad()
                        accelerator.backward(loss)
                        optimizer.step()

                        total_loss += loss.item()
                        tepoch.set_postfix(loss=total_loss / (step + 1))

        # Validation loss for early stopping.
        txt.eval()
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(enumerate(val_loader), unit="batch",
                      total=len(val_loader), dynamic_ncols=True) as tepoch:
                for step, data in tepoch:
                    tepoch.set_description("Val")
                    labels = data["labels"].to(device).view(-1, num_labels)
                    _, logits = txt(data, device)
                    eval_loss += reg_fct(logits.view(-1, num_labels), labels).item()
                    tepoch.set_postfix(loss=eval_loss / (step + 1))
        return eval_loss / len(val_loader)

    def save_checkpoint():
        if not accelerator.is_main_process:
            return
        os.makedirs(os.path.dirname(args.text_ckpt) or ".", exist_ok=True)
        torch.save(accelerator.unwrap_model(txt).state_dict(), args.text_ckpt)

    def evaluate():
        txt.eval()
        preds, targets = [], []
        with torch.no_grad():
            with tqdm(enumerate(testing_loader), unit="batch",
                      total=len(testing_loader), dynamic_ncols=True) as tepoch:
                for _, data in tepoch:
                    tepoch.set_description("Test")
                    _, logits = txt(data, device)
                    preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
                    # labels arrive as (B, 1, num_labels); flatten to one target per example
                    targets.extend(
                        torch.argmax(data["labels"], dim=-1).view(-1).cpu().numpy().tolist())
        return preds, targets

    if args.do_train:
        min_loss = float("inf")
        patience = EARLY_STOP_PATIENCE
        for epoch in range(args.epochs):
            val_loss = train_one_epoch(epoch)
            if val_loss < min_loss:
                min_loss = val_loss
                patience = EARLY_STOP_PATIENCE
                save_checkpoint()  # keep the best model
            elif patience > 0:
                patience -= 1
            else:
                print(f"early stop (best val loss {min_loss:.4f})")
                break
        # Reload the best checkpoint before evaluating.
        if os.path.exists(args.text_ckpt):
            accelerator.unwrap_model(txt).load_state_dict(
                torch.load(args.text_ckpt, weights_only=True), strict=False)

    if args.do_eval:
        preds, targets = evaluate()
        preds, targets = np.array(preds), np.array(targets)
        print(f"WA  = {weighted_accuracy(list_y_true=targets, list_y_pred=preds)}")
        print(f"UA  = {unweighted_accuracy(list_y_true=targets, list_y_pred=preds)}")
        print(f"UAR = {uar_score(labels=targets, predictions=preds)}")

    logger.flush()
    logger.close()


if __name__ == "__main__":
    main()
