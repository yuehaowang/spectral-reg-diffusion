#!/usr/bin/env bash
# Evaluate a baseline ImageNet-64 model: sample and score against a reference batch.
set -euo pipefail

# ---- edit these ----
NGPU=8
DATA=/path/to/imagenet-64x64.zip            # same archive used for training
OUTPUT=/path/to/dit_output                  # must contain $EXPNAME/checkpoints/
EXPNAME=imagenet64_baseline
REF=/path/to/VIRTUAL_imagenet64_labeled.npz # reference batch
CKPT_ITERS=100000                           # checkpoint iteration to evaluate
# --------------------

torchrun --standalone --nproc_per_node="$NGPU" generate.py \
    --base_dir "$OUTPUT" --expname "$EXPNAME" \
    --dataset_path "$DATA" --dataset_type zip --image_size 64 \
    --model_type dit-xl4 --cond \
    --reload_ema --no_eval_uncond \
    --eval_sample_num 50000 --eval_batch 256 --sampler_max_steps 50 \
    --eval_ckpt_iters $CKPT_ITERS \
    --compute_metrics --ref_batch_path "$REF"
