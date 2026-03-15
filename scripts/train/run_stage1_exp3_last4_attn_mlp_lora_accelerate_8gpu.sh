#!/usr/bin/env bash
set -euo pipefail

export RCCL_DEBUG=INFO
export ROCM_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRANSFORMERS_VERBOSITY=error
export MIOPEN_LOG_LEVEL=1
export MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK=1

export WANDB_API_KEY=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
export WANDB_PROJECT=ifdl_stage1_magicbrush
export WANDB_NAME=stage1_exp3_last4_attn_mlp_lora

accelerate launch \
  --multi_gpu \
  --num_processes 8 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port 29518 \
  scripts/train_stage1.py \
  --config configs/experiments/stage1_exp3_last4_attn_mlp_lora.yaml
