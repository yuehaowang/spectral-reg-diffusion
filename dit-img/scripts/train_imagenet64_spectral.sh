#!/usr/bin/env bash
# Ours: rectified-flow DiT + spectral representation regularization on ImageNet-64.
set -euo pipefail

# ---- edit these ----
NGPU=8
DATA=/path/to/imagenet-64x64.zip   # EDM-style zip archive (images + dataset.json labels)
OUTPUT=/path/to/dit_output         # run is written to $OUTPUT/$EXPNAME
EXPNAME=imagenet64_spectral
# --------------------

torchrun --standalone --nproc_per_node="$NGPU" train.py \
    --base_dir "$OUTPUT" --expname "$EXPNAME" --seed 0 \
    --dataset_path "$DATA" --dataset_type zip --image_size 64 \
    --model_type dit-xl4 --cond \
    --batch 64 --lr 1e-4 --max_iters 100000 \
    --ssrepl --enc_depth 8 --proj_bn --proj_adaln \
    --ssrepl_alpha 0.05 --ssrepl_lambda 0.08 --ssrepl_global_batch \
    --ckpt_every 10000 --eval_sample_every 8000 --eval_fixed_x0 \
    --sampler_max_steps 50 --log_every 100
