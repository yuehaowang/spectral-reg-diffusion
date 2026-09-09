#!/usr/bin/env bash
# Baseline rectified-flow DiT on FFHQ (64x64, unconditional), 70k iters.
# batch = 256 = 2x the ss-DiT batch, to match ss-DiT's two-view (2x) forward compute.
set -euo pipefail

# ---- edit these ----
NGPU=8
DATA=/path/to/ffhq-64x64.zip   # EDM-style zip archive
OUTPUT=/path/to/dit_output     # run is written to $OUTPUT/$EXPNAME
EXPNAME=ffhq_baseline
# --------------------

torchrun --standalone --nproc_per_node="$NGPU" train.py \
    --base_dir "$OUTPUT" --expname "$EXPNAME" --seed 0 \
    --dataset_path "$DATA" --dataset_type zip --image_size 64 \
    --model_type dit-s \
    --batch 256 --lr 1e-4 --max_iters 70000 \
    --ckpt_every 10000 --eval_sample_every 8000 --eval_fixed_x0 \
    --sampler_max_steps 50 --log_every 100
