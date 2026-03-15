#!/usr/bin/env bash
set -euo pipefail

python scripts/eval_stage1.py \
  --config configs/experiments/stage1_exp2_last4_attn_lora.yaml \
  --split test
