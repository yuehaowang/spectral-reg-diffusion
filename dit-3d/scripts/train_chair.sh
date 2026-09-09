#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIT3D_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DIT3D_DIR}"

exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    "${PYTHON_BIN:-python}" train.py \
    --dataroot "${DATA_ROOT:-${DIT3D_DIR}/ShapeNetCore.v2.PC15k}" \
    --category chair \
    --model_dir "${OUTPUT_ROOT:-${DIT3D_DIR}/output}" \
    --experiment_name "${EXPERIMENT_NAME:-ditw_S4_chair}" \
    --model_type 'DiT-S/4' \
    --window_size 4 \
    --window_block_indexes '0,3,6,9' \
    --bs "${BATCH_SIZE:-1024}" \
    --niter "${NITER:-10000}" \
    --voxel_size 32 \
    --beta_start 0.0001 \
    --beta_end 0.02 \
    --schedule_type linear \
    --time_num 1000 \
    --lr "${LEARNING_RATE:-1e-4}" \
    --saveIter "${SAVE_ITER:-100}" \
    --vizIter "${VIZ_ITER:-100}" \
    --print_freq "${PRINT_FREQ:-1}" \
    --manualSeed "${MANUAL_SEED:-42}" \
    --dist_backend "${DIST_BACKEND:-nccl}" \
    --port "${PORT:-12345}" \
    --use_tb \
    "$@"
