#!/usr/bin/env bash
# Ours: rectified-flow DiT + spectral representation regularization on CelebA
# (32x32, class-conditional), 70k iters. Uses --ssrepl_single_t (one shared timestep
# across the two views), the decisive setting on CelebA (23.38 vs 27.99 without).
set -euo pipefail

# ---- edit these ----
NGPU=8
DATA=/path/to/celeba_root      # dir containing celeba/img_align_celeba.zip + metadata
OUTPUT=/path/to/dit_output     # run is written to $OUTPUT/$EXPNAME
EXPNAME=celeba_spectral
# --------------------

torchrun --standalone --nproc_per_node="$NGPU" train.py \
    --base_dir "$OUTPUT" --expname "$EXPNAME" --seed 0 \
    --dataset_path "$DATA" --dataset_type celeba --image_size 32 \
    --model_type dit-s --cond \
    --batch 512 --lr 1e-4 --max_iters 70000 \
    --ssrepl --enc_depth 5 --proj_bn --proj_adaln \
    --ssrepl_alpha 0.05 --ssrepl_lambda 0.08 --ssrepl_global_batch --ssrepl_single_t \
    --ckpt_every 10000 --eval_sample_every 8000 --eval_fixed_x0 \
    --sampler_max_steps 50 --log_every 100
