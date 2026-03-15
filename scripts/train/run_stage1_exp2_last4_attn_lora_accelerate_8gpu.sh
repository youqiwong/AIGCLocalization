#!/usr/bin/env bash
set -euo pipefail

export RCCL_DEBUG=INFO
export ROCM_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRANSFORMERS_VERBOSITY=error
export MIOPEN_LOG_LEVEL=1
export MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK=1

: "${WANDB_API_KEY:?Please export WANDB_API_KEY before running.}"
export WANDB_PROJECT="${WANDB_PROJECT:-aigc_stage1_magicbrush}"
export WANDB_NAME="${WANDB_NAME:-exp2_last4_attn_lora}"

accelerate launch --num_processes 8 --num_machines 1 \
  scripts/train_stage1.py \
  --config configs/experiments/stage1_exp2_last4_attn_lora.yaml

python scripts/eval_stage1.py \
  --config configs/experiments/stage1_exp2_last4_attn_lora.yaml \
  --split test
