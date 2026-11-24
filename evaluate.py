"""Evaluate a trained checkpoint on the test (or dev) split — no training.

Pick the model with ``--stage`` and point ``--ckpt`` at the checkpoint to score
(defaults to that stage's standard training output). Prints WA / UA / UAR,
per-class recall and the confusion matrix.

The stage selects the matching output head:

    text              text-encoder logits              (text_script.py output)
    align / infogan   audio-only logits                (audio_script.py output)
    fusion            audio-text fused logits          (audio_script.py output;
                                                        also needs --text-ckpt)

Examples
--------
    python evaluate.py --stage text
    python evaluate.py --stage align
    python evaluate.py --stage fusion --ckpt checkpoints/audio_fusion.pt \
                       --text-ckpt checkpoints/text.pt
"""

import argparse
import os
import warnings
from contextlib import nullcontext

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

import common
from measure_WA_UA import unweighted_accuracy, uar_score, weighted_accuracy

warnings.filterwarnings("ignore")

# Default checkpoint to score for each stage (that stage's training output).
STAGE_CKPTS = {
    "text": "checkpoints/text.pt",
    "align": "checkpoints/audio_align.pt",
    "infogan": "checkpoints/audio_infogan.pt",
    "fusion": "checkpoints/audio_fusion.pt",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint (test or dev split, no training).")
    common.add_common_args(parser, train=False)
    parser.add_argument(
        "--stage", required=True, choices=["text", "align", "infogan", "fusion"],
        help="Which model to evaluate; selects the matching output head.")
    parser.add_argument(
        "--ckpt", default=None,
        help="Checkpoint to score (default: the stage's training output path).")
    parser.add_argument(
        "--text-ckpt", default="checkpoints/text.pt",
        help="Stage-1 text encoder; only used (and required) by the fusion head.")
    parser.add_argument(
        "--split", default="test", choices=["test", "dev"],
        help="Which split to evaluate (default: test).")
    args = parser.parse_args()

    if args.ckpt is None:
        args.ckpt = STAGE_CKPTS[args.stage]
    if args.batch_size is None:
        args.batch_size = 128 if args.stage == "text" else 8
    if args.data_root is None:
        args.data_root = common.DEFAULT_DATA_ROOT[args.dataset]

    if not os.path.exists(args.ckpt):
        parser.error(f"--ckpt '{args.ckpt}' not found; train that stage first.")
    if args.stage == "fusion" and not os.path.exists(args.text_ckpt):
        parser.error(
            f"--text-ckpt '{args.text_ckpt}' not found; "
            "the fusion head needs the Stage-1 text encoder.")
    return args


def predict(loader, forward, device):
    """Run ``forward`` over a loader; return argmax predictions and targets."""
    amp = (torch.autocast(device.type, dtype=torch.float16)
           if device.type == "cuda" else nullcontext())
    preds, targets = [], []
    with torch.no_grad():
        for data in tqdm(loader, unit="batch", dynamic_ncols=True, desc="Eval"):
            with amp:
                logits = forward(data)
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
            # labels arrive as (B, 1, num_labels); flatten to one target per example
            targets.extend(torch.argmax(data["labels"], dim=-1).view(-1).cpu().numpy().tolist())
    return np.array(preds), np.array(targets)


def eval_text(args, examples, num_labels, device):
    """Score the Stage-1 text encoder's own classification head."""
    from src.datacollator import TXTDataCollatorWithPadding
    from src.dataset import TXTDataset_aug

    txt, _, tokenizer = common.build_text_encoder(
        args.text_backbone, num_labels, device, freeze=True, ckpt=args.ckpt)
    txt.eval()

    dataset = TXTDataset_aug(list(examples))
    dataset.set_tokenizer(tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=TXTDataCollatorWithPadding(padding=True), num_workers=0)

    return predict(loader, lambda data: txt(data, device)[1], device)


def eval_audio(args, examples, num_labels, device):
    """Score an audio checkpoint: audio-only head, or the fused head for `fusion`."""
    from transformers import Wav2Vec2Config, Wav2Vec2Processor
    from src.datacollator import DataCollatorCTCWithPadding
    from src.dataset import Dataset_aug

    config_a = Wav2Vec2Config.from_pretrained(args.audio_backbone, num_labels=num_labels)
    config_a.proj_size = config_a.hidden_size
    processor = Wav2Vec2Processor.from_pretrained(args.audio_backbone)

    model = common.build_audio_encoder(
        config_a, num_labels, args.audio_backbone, ckpt=args.ckpt)
    model.to(device)
    model.eval()

    dataset = Dataset_aug(list(examples))

    # Only the fusion head consumes the (frozen) text encoder's features, so the
    # 7B text model is loaded — and transcripts tokenized — for that stage alone.
    if args.stage == "fusion":
        txt, _, tokenizer = common.build_text_encoder(
            args.text_backbone, num_labels, device, freeze=True, ckpt=args.text_ckpt)
        txt.eval()
        dataset.set_tokenizer(tokenizer)

        def forward(data):
            t, _ = txt(data, device)
            return model(data, device, t=t).logits_fuse
    else:
        def forward(data):
            return model(data, device).logits

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=DataCollatorCTCWithPadding(processor=processor, padding=True),
                        num_workers=0)

    return predict(loader, forward, device)


def report(preds, targets, class_names):
    """Print WA / UA / UAR, per-class recall and the confusion matrix."""
    print(f"\nWA  = {weighted_accuracy(list_y_true=targets, list_y_pred=preds)}")
    print(f"UA  = {unweighted_accuracy(list_y_true=targets, list_y_pred=preds)}")
    print(f"UAR = {uar_score(labels=targets, predictions=preds)}")

    n = len(class_names)
    cm = confusion_matrix(targets, preds, labels=list(range(n)))
    support = cm.sum(axis=1)
    name_w = max(len(c) for c in class_names)
    cell_w = max(name_w, len(str(cm.max()))) + 2

    print("\nPer-class recall:")
    for i, name in enumerate(class_names):
        recall = cm[i, i] / support[i] if support[i] else float("nan")
        print(f"  {name:<{name_w}}  {recall:.4f}  (n={support[i]})")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(" " * (name_w + 2) + "".join(f"{c:>{cell_w}}" for c in class_names))
    for i, name in enumerate(class_names):
        print(f"  {name:<{name_w}}" + "".join(f"{cm[i, j]:>{cell_w}}" for j in range(n)))


def main():
    args = parse_args()
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = common.build_dataset(args.dataset, args.data_root, args.refresh_cache)
    num_labels = dataset.num_labels
    _, dev_examples, test_examples = common.resolve_splits(dataset, args.fold, lim=args.lim)
    examples = test_examples if args.split == "test" else dev_examples

    # Class names for the per-class report (first name wins if several share an id).
    id2label = {}
    for name, idx in dataset.label_list.items():
        id2label.setdefault(idx, name)
    class_names = [id2label.get(i, str(i)) for i in range(num_labels)]

    if args.stage == "text":
        preds, targets = eval_text(args, examples, num_labels, device)
    else:
        preds, targets = eval_audio(args, examples, num_labels, device)

    print(f"\n[{args.stage}] {args.ckpt} on {args.dataset} {args.split} "
          f"({len(targets)} examples)")
    report(preds, targets, class_names)


if __name__ == "__main__":
    main()
