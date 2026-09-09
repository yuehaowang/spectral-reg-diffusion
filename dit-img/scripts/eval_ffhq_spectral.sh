#!/usr/bin/env bash
# Evaluate an "ours" FFHQ model (unconditional, spectral representation regularization):
# sample and score against a reference batch.
set -euo pipefail

# ---- edit these ----
NGPU=8
DATA=/path/to/ffhq-64x64.zip                # same archive used for training
OUTPUT=/path/to/dit_output                  # must contain $EXPNAME/checkpoints/
EXPNAME=ffhq_spectral
REF=/path/to/ref_ffhq_64_10000.npz          # reference batch
CKPT_ITERS=70000                            # checkpoint iteration to evaluate
# --------------------

torchrun --standalone --nproc_per_node="$NGPU" generate.py \
    --base_dir "$OUTPUT" --expname "$EXPNAME" \
    --dataset_path "$DATA" --dataset_type zip --image_size 64 \
    --model_type dit-s \
    --ssrepl --enc_depth 5 --proj_bn --proj_adaln \
    --reload_ema \
    --eval_sample_num 50000 --eval_batch 256 --sampler_max_steps 50 \
    --eval_ckpt_iters $CKPT_ITERS \
    --compute_metrics --ref_batch_path "$REF"
