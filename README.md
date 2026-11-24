# Improving Speech Emotion Recognition with Mutual Information Regularized Generative Model

Lack of large, well- annotated emotional speech corpora continues to limit the performance and robustness of speech emotion recognition (SER), particularly as models grow more complex and the demand for multimodal systems increases. While generative data augmentation offers a promising solution, existing approaches often produce emotionally inconsistent samples due to oversimplified conditioning on categorical labels. This paper introduces a novel mutual-information-regularised generative framework that combines cross-modal alignment with feature-level synthesis.
Building on an InfoGAN-style architecture, our method first learns a semantically aligned audio–text representation space using pre-trained transformers and contrastive objectives.
A feature generator is then trained to produce emotion-aware audio features while employing mutual information as a quantitative regulariser to ensure strong dependency between generated features and their conditioning variables. We extend this approach to multimodal settings, enabling the generation of novel, paired (audio, text) features. Comprehensive evaluation on three benchmark datasets (IEMOCAP, MSP-IMPROV, MSP-Podcast) demonstrates that our framework consistently outperforms existing augmentation methods, achieving state-of-the-art performance with improvements of up to 2.6\% in unimodal SER and 3.2\% in multimodal emotion recognition. Most importantly, we demonstrate that mutual information functions as both a regulariser and a measurable metric for generative quality, offering a systematic approach to data augmentation in affective computing.

## Pipeline

The model is trained in four sequential stages. Each stage loads the checkpoint(s)
produced by the previous one.

| Stage | Command | What it does | Output |
|------:|---------|--------------|--------|
| 1 | `text_script.py` | Train the text transformer (emotion from text). | `text.pt` |
| 2a | `audio_script.py --stage align` | Train the audio transformer, aligning audio to the frozen text encoder (cross-entropy + two InfoNCE alignment losses). | `audio_align.pt` |
| 2b | `audio_script.py --stage infogan` | Train an InfoGAN that generates audio features before the classification head (D / G / Q steps). | `audio_infogan.pt` |
| 2c | `audio_script.py --stage fusion` | Data-augment with generated audio/text features and train the audio–text fusion classifier. | `audio_fusion.pt` |

Throughout Stage 2 the text encoder from Stage 1 is loaded **frozen**.

## Repository layout

```
.
├── text_script.py        # Stage 1 entry point (text transformer)
├── audio_script.py       # Stage 2 entry point (--stage align|infogan|fusion)
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
python text_script.py --dataset msp-podcast --text-ckpt checkpoints/text.pt

# Stage 2 — audio–text model
python audio_script.py --stage align   --text-ckpt checkpoints/text.pt
python audio_script.py --stage infogan --text-ckpt checkpoints/text.pt
python audio_script.py --stage fusion  --text-ckpt checkpoints/text.pt
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

### Stage 1 (`text_script.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--text-ckpt` | `checkpoints/text.pt` | Save path (and resume path if it exists) |
| `--lr` | `1e-5` | Learning rate |
| `--do-train` / `--do-eval` | both | Restrict to training or evaluation only |

### Stage 2 (`audio_script.py`)

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

## Analysis notebooks

Companion notebooks for the paper's analyses and figures. Run them from the
repository root; each starts with an *Arguments* cell to edit (checkpoint paths,
dataset root, GPU). Figures are written to `figures/` as PDF + PNG.

| Notebook | What it does | Needs GPU |
|----------|--------------|:---------:|
| `validate_alignment.ipynb` | Audio–text alignment of a fusion checkpoint: MMD and paired statistics for matched vs shuffled pairs, plus fusion recognition with swapped text. Appends a row to `alignment_results.csv`. | yes |
| `validate_generation.ipynb` | Real vs InfoGAN-generated audio features (label + matched text vs label + random vector): MMD / cosine tables and PCA / t-SNE plots. Appends a row to `generation_results.csv`. | yes |
| `visualize_generation.ipynb` | The paper's t-SNE figure: real vs generated audio features on a shared embedding, coloured by emotion class (generated / real / overlay). Does not need the text model by default. | yes |
| `plot_generation_results.ipynb` | Box plots and a summary table across checkpoints from `generation_results.csv`. | no |

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
