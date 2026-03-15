#!/usr/bin/env bash
set -euo pipefail

python scripts/eval_stage1.py \
  --config configs/experiments/stage1_exp4_allblocks_attn_mlp_lora.yaml \
  --split test
