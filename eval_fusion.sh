#!/bin/bash
# Evaluate the Stage-2c (fusion) model on the MSP-PODCAST test set.
set -e

cd "$(dirname "$0")"   # run from the repo root

python evaluate.py \
    --stage fusion \
    --dataset msp-podcast \
    --data-root ../MSP-PODCAST-Publish-1.10/ \
    --split test \
    --ckpt ../F_Pllama_bal_gan_augmented_0.pt \
    --text-ckpt ../F_P_llama20k_0.pt \
    --batch-size 8