# Audio–Text Emotion Recognition with Text-Aligned Audio Augmentation

Speech emotion recognition that transfers knowledge from a text emotion model into
the audio branch. A text transformer is trained first, then a Wav2Vec2 audio encoder
is aligned to it, an InfoGAN learns to generate audio features, and finally an
audio–text fusion classifier is trained with the generated features as augmentation.

## Pipeline

The model is trained in four sequential stages. Each stage loads the checkpoint(s)
produced by the previous one.

| Stage | Command | What it does | Output |
|------:|---------|--------------|--------|
| 1 | `bert_script.py` | Train the text transformer (emotion from text). | `text.pt` |
| 2a | `aae_script.py --stage align` | Train the audio transformer, aligning audio to the frozen text encoder (cross-entropy + two InfoNCE alignment losses). | `audio_align.pt` |
| 2b | `aae_script.py --stage infogan` | Train an InfoGAN that generates audio features before the classification head (D / G / Q steps). | `audio_infogan.pt` |
| 2c | `aae_script.py --stage fusion` | Data-augment with generated audio/text features and train the audio–text fusion classifier. | `audio_fusion.pt` |

Throughout Stage 2 the text encoder from Stage 1 is loaded **frozen**.

## Repository layout

```
.
├── bert_script.py        # Stage 1 entry point (text transformer)
├── aae_script.py         # Stage 2 entry point (--stage align|infogan|fusion)
├── evaluate.py           # Evaluation-only entry point (any stage's checkpoint)
├── common.py             # Shared config, dataset dispatch, encoders, loss weights
├── measure_WA_UA.py      # WA / UA / UAR metrics
├── requirements.txt
└── src/
    ├── dataloadsplit.py  # IEMOCAP / MSP-IMPROV / MSP-PODCAST loaders
    ├── dataset.py        # Torch datasets (audio+text / text-only)
    ├── datacollator.py   # Batch collators
    ├── aug_bal.py        # Augmentation-balancing loss (fusion stage)
    ├── bertemb.py        # Text encoder (BERTEnc)
    ├── model__gan.py     # Audio encoder + InfoGAN (AudioEnc)
    ├── vsgan.py          # Generator / Discriminator
    ├── MMD.py            # Maximum Mean Discrepancy
    └── wav2vecdim.py     # Wav2Vec2 dimensional-emotion backbone
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable GPU is required for training. The text backbone defaults to
`meta-llama/Llama-2-7b-hf`, which is gated on the Hugging Face Hub — request access
and run `huggingface-cli login` before the first run.

## Data

Point the scripts at your dataset root with `--data-root` (defaults shown below).
`--dataset msp-podcast` (default) uses fixed train/dev/test splits; `iemocap` and
`msp-improv` use cross-validation folds, where `--fold N` selects the held-out test
fold (the next fold is used for validation).

| `--dataset` | default `--data-root` | splitting |
|-------------|-----------------------|-----------|
| `msp-podcast` | `../MSP-PODCAST` | fixed train/dev/test |
| `iemocap` | `IEMOCAP_full_release` | CV folds (`--fold`) |
| `msp-improv` | `../MSP-IMPROV` | CV folds (`--fold`) |

### Dataset cache

Parsing a corpus reads one transcript file per utterance, which can take minutes.
The parsed dataset is therefore pickled to `cache/` on first use and reloaded
(near-instantly) on every later run. This also freezes the IEMOCAP / MSP-IMPROV
cross-validation fold assignment, so training and any later evaluation are
guaranteed to see identical splits. Pass `--refresh-cache` (or delete the file
in `cache/`) to force a re-parse — e.g. after the corpus itself changed.

## Usage

Run the stages in order. Checkpoints are written to `checkpoints/` by default and
the flow is wired automatically (each Stage-2 sub-stage reads the previous output).

```bash
# Stage 1 — text transformer
python bert_script.py --dataset msp-podcast --text-ckpt checkpoints/text.pt

# Stage 2 — audio–text model
python aae_script.py --stage align   --text-ckpt checkpoints/text.pt
python aae_script.py --stage infogan --text-ckpt checkpoints/text.pt
python aae_script.py --stage fusion  --text-ckpt checkpoints/text.pt
```

Each script prints weighted accuracy (WA), unweighted accuracy (UA) and unweighted
average recall (UAR) on the test set, and logs training curves to TensorBoard
(`--log-dir`, default `logs/`).

### Common options (both scripts)

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `msp-podcast` | `iemocap` \| `msp-improv` \| `msp-podcast` |
| `--data-root` | per-dataset | Dataset root directory |
| `--fold` | `0` | Held-out test fold (IEMOCAP / MSP-IMPROV only) |
| `--epochs` | `100` (Stage 1), `10` (Stage 2) | Training epochs |
| `--batch-size` | `128` (Stage 1), `8` (Stage 2) | Mini-batch size |
| `--refresh-cache` | off | Re-parse the dataset instead of using `cache/` |
| `--lim` | off | Keep only the first N examples per split (smoke test, e.g. `--lim 10`) |
| `--text-backbone` | `meta-llama/Llama-2-7b-hf` | Pretrained text weights (Llama, BERT or RoBERTa) |
| `--audio-backbone` | `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | Pretrained audio weights |
| `--log-dir` | `logs` | TensorBoard directory |

### Stage 1 (`bert_script.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--text-ckpt` | `checkpoints/text.pt` | Save path (and resume path if it exists) |
| `--lr` | `1e-5` | Learning rate |
| `--do-train` / `--do-eval` | both | Restrict to training or evaluation only |

### Stage 2 (`aae_script.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--stage` | *(required)* | `align` \| `infogan` \| `fusion` |
| `--text-ckpt` | `checkpoints/text.pt` | Frozen Stage-1 text encoder (must exist) |
| `--audio-ckpt-in` | previous stage's output | Audio checkpoint to resume from |
| `--audio-ckpt-out` | per-stage path | Where to save this stage's audio model |

## Evaluation

`evaluate.py` scores an existing checkpoint without any training. `--stage` picks
the model and its output head; `--ckpt` defaults to that stage's standard training
output.

```bash
python evaluate.py --stage text      # scores checkpoints/text.pt
python evaluate.py --stage align     # scores checkpoints/audio_align.pt
python evaluate.py --stage infogan   # scores checkpoints/audio_infogan.pt
python evaluate.py --stage fusion    # scores checkpoints/audio_fusion.pt (+ --text-ckpt)
```

It prints WA / UA / UAR plus per-class recall and the confusion matrix.

| Flag | Default | Description |
|------|---------|-------------|
| `--stage` | *(required)* | `text` \| `align` \| `infogan` \| `fusion` |
| `--ckpt` | per-stage path | Checkpoint to score |
| `--text-ckpt` | `checkpoints/text.pt` | Stage-1 text encoder (fusion only) |
| `--split` | `test` | Evaluate the `test` or `dev` split |

Only the `text` and `fusion` stages load the 7B text model; `align` and `infogan`
evaluate the audio branch alone.

## Notes

- `--text-backbone` accepts Llama, BERT or RoBERTa checkpoints (e.g.
  `meta-llama/Llama-2-7b-hf`, `bert-base-uncased`, `roberta-large`); the
  architecture is detected from the backbone's config. Llama backbones are
  LoRA-wrapped with last-token pooling (the reported setup); BERT / RoBERTa are
  fully fine-tuned with [CLS] pooling. Use the same backbone for every stage —
  the Stage-2 scripts and `evaluate.py` must rebuild the text encoder that
  `--text-ckpt` was trained with.
- The audio architecture is fixed to Wav2Vec2; `--audio-backbone` swaps the
  pretrained weights, not the architecture.
- The `fusion` stage trains for the full number of epochs (it has no validation
  head); `align` and `infogan` use validation-loss early stopping.
- Cross-entropy uses inverse-frequency class weights for MSP-PODCAST (4-class) and
  is unweighted otherwise.
