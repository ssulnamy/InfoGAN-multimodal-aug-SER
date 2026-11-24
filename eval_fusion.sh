#!/bin/bash
# Evaluate the Stage-2c (fusion) model on the MSP-PODCAST test set.
# Uses the repository defaults: checkpoints/audio_fusion.pt scored with the
# Stage-1 text encoder checkpoints/text.pt. Pass extra evaluate.py flags to
# override, e.g. ./eval_fusion.sh --data-root /path/to/MSP-PODCAST --split dev
set -e

cd "$(dirname "$0")"   # run from the repo root

python evaluate.py \
    --stage fusion \
    --dataset msp-podcast \
    --ckpt checkpoints/audio_fusion.pt \
    --text-ckpt checkpoints/text.pt \
    --split test \
    --batch-size 8 \
    "$@"
