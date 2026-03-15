#!/usr/bin/env bash
set -euo pipefail

python scripts/eval_stage1.py \
  --config configs/experiments/stage1_exp3_last4_attn_mlp_lora.yaml \
  --split test
