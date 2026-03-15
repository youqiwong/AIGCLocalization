#!/usr/bin/env bash
set -euo pipefail

python scripts/train_stage1.py \
  --config configs/experiments/stage1_exp1_baseline_no_lora.yaml

python scripts/eval_stage1.py \
  --config configs/experiments/stage1_exp1_baseline_no_lora.yaml \
  --split test
