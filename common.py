"""Shared configuration and helpers for the audio-text emotion recognition pipeline.

Pipeline overview
-----------------
    Stage 1   bert_script.py                    -> text transformer (emotion from text)
    Stage 2a  aae_script.py --stage align       -> audio transformer aligned to text
    Stage 2b  aae_script.py --stage infogan     -> InfoGAN audio-feature generator
    Stage 2c  aae_script.py --stage fusion      -> audio-text fusion classifier

All stages share the same dataset loaders, backbones and loss weighting, which
live here so the two entry scripts stay small and consistent.
"""

import argparse
import os
import pickle

import torch

from src.dataloadsplit import IEMOCAPLoader, MSPIMPROVLoader, MSPPODCASTLoader

# Default pretrained backbones (override on the command line if needed).
# The architectures are fixed (Wav2Vec2 for audio, Llama for text); the flags
# only swap the specific pretrained weights of those architectures.
DEFAULT_AUDIO_BACKBONE = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
DEFAULT_TEXT_BACKBONE = "meta-llama/Llama-2-7b-hf"
SAMPLE_RATE = 16000
TEXT_FEATURE_DIM = 1024  # c_dim: dimensionality of the text conditioning vector

# Default dataset root directories (relative to the project root).
DEFAULT_DATA_ROOT = {
    "iemocap": "IEMOCAP_full_release",
    "msp-improv": "../MSP-IMPROV",
    "msp-podcast": "../MSP-PODCAST",
}

# Parsed-dataset cache. Parsing reads one transcript file per utterance, which is
# slow, so the parsed loader is pickled here on first use. The filenames encode
# the fixed parse parameters used in build_dataset(); if you change those
# parameters, change the name too (or pass --refresh-cache).
CACHE_DIR = "cache"
_CACHE_NAMES = {
    "iemocap": "iemocap_4class.pkl",
    "msp-improv": "msp-improv.pkl",
    "msp-podcast": "msp-podcast_min5.pkl",
}

# Inverse-frequency class weights used for the reported MSP-PODCAST 4-class runs
# (weight_c = N / (K * count_c)). Applied only for that dataset/label count; other
# datasets fall back to an unweighted loss.
_MSP_PODCAST_CE_WEIGHTS = [
    67749 / (4 * 37427),
    67749 / (4 * 20554),
    67749 / (4 * 5867),
    67749 / (4 * 4081),
]


def add_common_args(parser, train=True):
    """Register the CLI arguments shared by every stage.

    ``train=False`` skips the training-only flags (for evaluation scripts).
    """
    parser.add_argument(
        "--dataset", choices=["iemocap", "msp-improv", "msp-podcast"],
        default="msp-podcast", help="Which dataset to train/evaluate on.")
    parser.add_argument(
        "--data-root", default=None,
        help="Dataset root directory (default: per-dataset path in common.DEFAULT_DATA_ROOT).")
    parser.add_argument(
        "--fold", type=int, default=0,
        help="Held-out test fold for IEMOCAP / MSP-IMPROV (ignored for MSP-PODCAST).")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size.")
    parser.add_argument(
        "--refresh-cache", action="store_true",
        help="Re-parse the dataset instead of loading the pickle in cache/.")
    parser.add_argument(
        "--lim", type=int, default=None,
        help="Keep only the first N examples of each split (smoke test: "
             "e.g. --lim 10 to check a script runs end to end).")
    parser.add_argument("--text-backbone", default=DEFAULT_TEXT_BACKBONE)
    parser.add_argument("--audio-backbone", default=DEFAULT_AUDIO_BACKBONE)
    if train:
        parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
        parser.add_argument("--log-dir", default="logs", help="TensorBoard log directory.")
    return parser


def build_dataset(name, root, refresh_cache=False):
    """Instantiate a dataset loader by name, using the parse cache when possible.

    The parsed loader is pickled to ``cache/`` on first use and reloaded on later
    runs. Besides skipping the slow per-utterance transcript reads, this freezes
    the IEMOCAP / MSP-IMPROV cross-validation fold assignment, so training and
    any later evaluation see identical splits.
    """
    cache_path = os.path.join(CACHE_DIR, _CACHE_NAMES[name])
    if not refresh_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            dataset = pickle.load(f)
        if dataset.root_dir == root:
            print(f"loaded dataset cache: {cache_path}")
            return dataset
        print(f"dataset cache {cache_path} was built from '{dataset.root_dir}', "
              f"not '{root}'; re-parsing")

    if name == "iemocap":
        dataset = IEMOCAPLoader(root, fourclass=True)
    elif name == "msp-improv":
        dataset = MSPIMPROVLoader(root)
    elif name == "msp-podcast":
        # min=5 drops training utterances with <= 5 transcript words (as in the
        # reported runs); dev/test are kept complete.
        dataset = MSPPODCASTLoader(root, min=5)
    else:
        raise ValueError(f"unknown dataset '{name}'")

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"cached parsed dataset to {cache_path}")
    return dataset


def resolve_splits(dataset, fold=0, lim=None):
    """Return (train, dev, test) example lists for the given dataset.

    MSP-PODCAST provides fixed train/dev/test splits. IEMOCAP and MSP-IMPROV
    provide cross-validation folds in ``dataset.splits``; there we use ``fold``
    as the held-out test fold and the following fold as validation.

    ``lim`` truncates every split to its first ``lim`` examples. It is applied
    here, after the (full) dataset is parsed and cached, so a smoke-test run
    never poisons the cache.
    """
    if getattr(dataset, "train", None):
        train, dev, test = dataset.train, dataset.dev, dataset.test
    else:
        splits = list(dataset.splits)
        n = len(splits)
        test_idx = fold % n
        dev_idx = (fold + 1) % n
        test = splits[test_idx]
        dev = splits[dev_idx]
        train = [item for i, s in enumerate(splits)
                 if i not in (test_idx, dev_idx) for item in s]

    if lim is not None:
        train, dev, test = train[:lim], dev[:lim], test[:lim]
        print(f"--lim {lim}: splits truncated to "
              f"{len(train)}/{len(dev)}/{len(test)} train/dev/test examples")
    return train, dev, test


def class_weights(dataset_name, num_labels, device):
    """Cross-entropy class weights.

    Returns the exact reported inverse-frequency weights for MSP-PODCAST 4-class
    runs, and ``None`` (unweighted) otherwise.
    """
    if dataset_name == "msp-podcast" and num_labels == 4:
        return torch.tensor(_MSP_PODCAST_CE_WEIGHTS, device=device)
    return None


def build_text_encoder(backbone, num_labels, device, freeze=False, ckpt=None):
    """Build the text encoder (``BERTEnc``) around a pretrained backbone.

    The architecture is detected from the backbone's config: Llama backbones get
    a LoRA-wrapped causal LM with last-token pooling (the reported setup); BERT /
    RoBERTa backbones are fully fine-tuned encoders with [CLS] pooling.

    If ``ckpt`` points to an existing file the weights are loaded into a config-
    initialised model; otherwise the pretrained backbone is downloaded/loaded.
    """
    import os
    from transformers import AutoConfig
    from src.bertemb import BERTEnc

    config = AutoConfig.from_pretrained(
        backbone, num_labels=num_labels, finetuning_task="classification")
    load_ckpt = bool(ckpt) and os.path.exists(ckpt)

    if config.model_type == "llama":
        from transformers import LlamaForCausalLM, LlamaTokenizer
        from peft import LoraConfig, get_peft_model

        config.proj_size = config.hidden_size
        config.final_dropout = 0.5
        tokenizer = LlamaTokenizer.from_pretrained(backbone)
        tokenizer.pad_token = tokenizer.eos_token

        lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1)
        base = get_peft_model(
            LlamaForCausalLM(config=config) if load_ckpt
            else LlamaForCausalLM.from_pretrained(backbone), lora)
        txt = BERTEnc(model_t=base, config_t=config, freeze=freeze, pool="last")
        txt.model.gradient_checkpointing_enable()
    elif config.model_type in ("bert", "roberta"):
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(backbone)
        base = (AutoModel.from_config(config) if load_ckpt
                else AutoModel.from_pretrained(backbone, config=config))
        txt = BERTEnc(model_t=base, config_t=config, freeze=freeze, pool="cls")
    else:
        raise ValueError(
            f"unsupported text backbone architecture '{config.model_type}' "
            "(supported: llama, bert, roberta)")

    if load_ckpt:
        txt.load_state_dict(torch.load(ckpt, weights_only=True), strict=False)
        print(f"loaded text encoder from checkpoint: {ckpt}")
    else:
        print(f"initialised text encoder from pretrained backbone: {backbone}")

    txt.to(device)
    return txt, config, tokenizer


def build_audio_encoder(config_a, num_labels, backbone, ckpt=None, ce_weight=None):
    """Build the Wav2Vec2 + InfoGAN audio encoder (``AudioEnc``).

    If ``ckpt`` points to an existing file the weights are loaded into a config-
    initialised model; otherwise the pretrained backbone is downloaded/loaded.
    """
    import os
    from src.model__gan import AudioEnc
    from src.wav2vecdim import EmotionModel as Wav2Vec2ForAVD

    if ckpt and os.path.exists(ckpt):
        model = AudioEnc(model_a=Wav2Vec2ForAVD(config_a), config_a=config_a,
                         num_labels=num_labels, c_dim=TEXT_FEATURE_DIM, ce_weight=ce_weight)
        model.load_state_dict(torch.load(ckpt, weights_only=True), strict=False)
        print(f"loaded audio encoder from checkpoint: {ckpt}")
    else:
        base = Wav2Vec2ForAVD.from_pretrained(backbone)
        model = AudioEnc(model_a=base, config_a=config_a,
                         num_labels=num_labels, c_dim=TEXT_FEATURE_DIM, ce_weight=ce_weight)
        print(f"initialised audio encoder from pretrained backbone: {backbone}")
    return model
