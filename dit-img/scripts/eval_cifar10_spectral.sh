#!/usr/bin/env bash
# Evaluate an "ours" CIFAR-10 model (spectral representation regularization):
# sample and score against a reference batch.
set -euo pipefail

# ---- edit these ----
NGPU=8
DATA=/path/to/cifar10_data                  # same data dir used for training
OUTPUT=/path/to/dit_output                  # must contain $EXPNAME/checkpoints/
EXPNAME=cifar10_spectral
REF=/path/to/ref_edm_cifar10-32x32.npz      # reference batch
CKPT_ITERS=70000                            # checkpoint iteration to evaluate
# --------------------

torchrun --standalone --nproc_per_node="$NGPU" generate.py \
    --base_dir "$OUTPUT" --expname "$EXPNAME" \
    --dataset_path "$DATA" --dataset_type cifar10 --image_size 32 \
    --model_type dit-s --cond \
    --ssrepl --enc_depth 5 --proj_bn --proj_adaln \
    --reload_ema --no_eval_uncond \
    --eval_sample_num 50000 --eval_batch 512 --sampler_max_steps 50 \
    --eval_ckpt_iters $CKPT_ITERS \
    --compute_metrics --ref_batch_path "$REF"
