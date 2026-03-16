#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${1:-1}"
BATCH_SIZE="${2:-0}"
if (( NUM_GPUS < 1 || NUM_GPUS > 8 )); then
  echo "NUM_GPUS must be in [1, 8], got ${NUM_GPUS}" >&2
  exit 1
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_LIST="$(seq -s, 0 $((NUM_GPUS - 1)))"
  export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
fi

python scripts/eval_stage1.py \
  --config configs/experiments/stage1_exp1_baseline_no_lora.yaml \
  --split test \
  --eval-all-datasets \
  --num-gpus "${NUM_GPUS}" \
  --batch-size "${BATCH_SIZE}"
