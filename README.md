# Stage1 Forgery Perception Backbone for Open-World AIGC Inpainting Localization

This repository implements **Stage1** of a two-stage framework for AIGC inpainting localization.

Stage1 consumes a single RGB image and produces:
- image-level edit probability `p_edit`
- coarse suspicion heatmap `H`
- initial pixel mask `M0`
- forgery-sensitive multi-scale features
- top-k candidate regions derived from `H`

## Quick start

```bash
python scripts/inspect_magicbrush.py \
  --data-root /home/tione/notebook/datasets/MagicBrush/data

python scripts/build_magicbrush_manifest.py \
  --data-root /home/tione/notebook/datasets/MagicBrush/data \
  --output-dir artifacts/manifests \
  --mode debug

python scripts/inspect_qwen3vl_modules.py \
  --model-path /home/tione/notebook/models/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct \
  --only-linear

python scripts/train_stage1.py --config configs/stage1_debug.yaml
python scripts/eval_stage1.py --config configs/stage1_debug.yaml --split test
```

## Notes

- The parser does not assume parquet schema up front. It inspects columns and infers likely image/mask/edit-turn fields.
- Multi-turn edits are expanded into independent samples, and each source group also generates one clean sample.
- If a mask is missing for an edited turn, building the manifest raises an error with the sample id.
- Manifest stores parquet index references (path + row_group + row index + field names), not embedded base64 image bytes.
- Training uses tqdm progress bars for manifest loading, train steps, and val loops; default validation frequency is every 500 steps.
