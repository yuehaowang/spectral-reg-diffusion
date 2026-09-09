#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIT3D_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DIT3D_DIR}"

MODEL="${MODEL:?MODEL=<path-to-epoch_*.pth> required}"

exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "${PYTHON_BIN:-python}" eval.py \
    --ss \
    --dataroot "${DATA_ROOT:-${DIT3D_DIR}/ShapeNetCore.v2.PC15k}" \
    --category car \
    --eval-split "${EVAL_SPLIT:-val}" \
    --model_dir "${OUTPUT_ROOT:-${DIT3D_DIR}/eval_output}" \
    --experiment_name "${EXPERIMENT_NAME:-ssditw_S4_car_enc6_pcoeff0.05_tcoeff0.08_eval}" \
    --model_type 'DiT-S/4' \
    --window_size 4 \
    --window_block_indexes '0,3,6,9' \
    --proj-dims 512 256 \
    --proj-coeff 0.05 \
    --proj-triu-coeff 0.08 \
    --proj-adaln \
    --proj-bn \
    --encoder-depth 6 \
    --bs "${BATCH_SIZE:-32}" \
    --workers "${WORKERS:-8}" \
    --npoints 2048 \
    --voxel_size 32 \
    --beta_start 0.0001 \
    --beta_end 0.02 \
    --schedule_type linear \
    --time_num 1000 \
    --model "${MODEL}" \
    --metrics "${METRICS:-both}" \
    --metric-batch-size "${METRIC_BATCH_SIZE:-64}" \
    --metric-sample-batch-size "${METRIC_SAMPLE_BATCH_SIZE:-1}" \
    --max-eval-shapes "${MAX_EVAL_SHAPES:-0}" \
    --manualSeed "${MANUAL_SEED:-0}" \
    --no-rand-cls \
    --strict-dataset-counts \
    "$@"
