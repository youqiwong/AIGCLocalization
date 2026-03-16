#!/usr/bin/env python3
import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aigc_datasets.magicbrush_dataset import MagicBrushDataset, collate_magicbrush_batch
from aigc_datasets.ood_eval_dataset import OODEvalDataset
from losses import bce_dice_loss, detection_bce_loss, edge_bce_loss, focal_heatmap_loss
from models.stage1_model import Stage1ForgeryModel
from utils.checkpoint import load_checkpoint, load_stage1_checkpoint_into_model
from utils.metrics import binary_auc_ap, cls_metrics, pixel_metrics
from utils.vis import save_eval_annotated_vis


class EvalStage1Wrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor, out_hw):
        out = self.model(pixel_values, image_grid_thw, out_hw=out_hw)
        result = {
            "p_edit": out["p_edit"],
            "heatmap": out["heatmap"],
            "mask0": out["mask0"],
        }
        if "edge0" in out:
            result["edge0"] = out["edge0"]
        return result


def _forged_only_loss(loss_fn, pred: torch.Tensor, target: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, int]:
    forged = label >= 0.5
    forged_count = int(forged.sum().item())
    if forged_count == 0:
        return pred.float().sum() * 0.0, 0
    return loss_fn(pred[forged], target[forged]), forged_count


def resolve_eval_output_dir(base_output_dir: str) -> Path:
    root = Path(base_output_dir)
    if not root.exists():
        raise FileNotFoundError(f"output_dir does not exist: {root}")
    if (root / "best_by_iou.pt").exists() or (root / "best_by_iou_full.pt").exists():
        return root
    candidates = [p for p in root.iterdir() if p.is_dir() and ((p / "best_by_iou.pt").exists() or (p / "best_by_iou_full.pt").exists())]
    if not candidates:
        raise FileNotFoundError(f"no timestamped run directory with best_by_iou.pt found under: {root}")
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def build_loader(dataset, cfg: Dict[str, Any], batch_size: int) -> DataLoader:
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_magicbrush_batch,
    )
    if int(cfg["data"]["num_workers"]) > 0:
        loader_kwargs["prefetch_factor"] = int(cfg["data"].get("prefetch_factor", 1))
        loader_kwargs["persistent_workers"] = bool(cfg["data"].get("persistent_workers", True))
    return DataLoader(**loader_kwargs)


def evaluate_loader(
    *,
    dataset_name: str,
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    num_gpus: int,
    cfg: Dict[str, Any],
    vis_path: str,
    max_vis_items: int,
) -> Dict[str, Any]:
    model.eval()
    probs, labels, pred_masks, gt_masks = [], [], [], []
    loss_sums = {"l_det": 0.0, "l_heat": 0.0, "l_mask": 0.0, "l_edge": 0.0}
    total_items = 0
    total_forged_items = 0
    use_edge_loss = bool(cfg["train"].get("use_edge_loss", True))
    vis_saved = False
    print(
        json.dumps(
            {
                "dataset": dataset_name,
                "num_samples": len(loader.dataset),
                "num_batches": len(loader),
                "total_batch_size": loader.batch_size,
                "num_gpus": num_gpus,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval-{dataset_name}", dynamic_ncols=True):
            image = batch["image"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            image_grid_thw = batch["image_grid_thw"].to(device)
            label = batch["label"].to(device)
            mask = batch["mask"].to(device)
            edge_gt = batch["edge_gt"].to(device)
            out = model(pixel_values, image_grid_thw, out_hw=mask.shape[-2:])

            heat_target = torch.nn.functional.interpolate(mask, size=out["heatmap"].shape[-2:], mode="nearest")
            l_det = detection_bce_loss(out["p_edit"], label)
            l_heat, batch_forged_items = _forged_only_loss(focal_heatmap_loss, out["heatmap"], heat_target, label)
            l_mask, _ = _forged_only_loss(bce_dice_loss, out["mask0"], mask, label)
            if use_edge_loss and "edge0" in out:
                l_edge, _ = _forged_only_loss(edge_bce_loss, out["edge0"], edge_gt, label)
            else:
                l_edge = torch.zeros_like(l_det)
            loss = (
                float(cfg["train"]["lambda_det"]) * l_det
                + float(cfg["train"]["lambda_heat"]) * l_heat
                + float(cfg["train"]["lambda_mask"]) * l_mask
                + float(cfg["train"].get("lambda_edge", 0.1)) * l_edge
            )

            probs.append(out["p_edit"].cpu())
            labels.append(label.cpu())
            pred_masks.append(out["mask0"].cpu())
            gt_masks.append(mask.cpu())
            batch_items = int(label.shape[0])
            total_forged_items += int(batch_forged_items)
            total_items += batch_items
            loss_sums["l_det"] += float(l_det.item()) * batch_items
            loss_sums["l_heat"] += float(l_heat.item()) * batch_forged_items
            loss_sums["l_mask"] += float(l_mask.item()) * batch_forged_items
            loss_sums["l_edge"] += float(l_edge.item()) * batch_forged_items

            if not vis_saved:
                save_eval_annotated_vis(
                    image=image.cpu(),
                    gt_mask=mask.cpu(),
                    heatmap=out["heatmap"].cpu(),
                    pred_mask=out["mask0"].cpu(),
                    prob=out["p_edit"].cpu(),
                    label=label.cpu(),
                    path=vis_path,
                    max_items=max_vis_items,
                )
                vis_saved = True

    prob = torch.cat(probs, dim=0)
    label = torch.cat(labels, dim=0)
    pred_mask = torch.cat(pred_masks, dim=0)
    gt_mask = torch.cat(gt_masks, dim=0)

    forged = label >= 0.5
    real = ~forged
    cls = cls_metrics(prob, label)
    auc_ap = binary_auc_ap(prob, label)
    pixel_all = pixel_metrics(pred_mask, gt_mask)
    pixel_forged = pixel_metrics(pred_mask[forged], gt_mask[forged]) if forged.any() else {"f1": 0.0, "iou": 0.0}
    pred_cls = (prob >= 0.5).float()
    real_acc = float((pred_cls[real] == label[real]).float().mean().item()) if real.any() else 0.0
    forged_acc = float((pred_cls[forged] == label[forged]).float().mean().item()) if forged.any() else 0.0

    result = {
        "dataset": dataset_name,
        "auc": auc_ap["auc"],
        "ap": auc_ap["ap"],
        "image_f1": cls["f1"],
        "image_acc": cls["acc"],
        "image_precision": cls["precision"],
        "image_recall": cls["recall"],
        "pixel_f1": pixel_forged["f1"],
        "pixel_iou": pixel_forged["iou"],
        "pixel_f1_all": pixel_all["f1"],
        "pixel_iou_all": pixel_all["iou"],
        "num_samples": int(label.numel()),
        "num_real": int(real.sum().item()),
        "num_forged": int(forged.sum().item()),
        "real_acc": real_acc,
        "forged_acc": forged_acc,
    }
    result["l_det"] = loss_sums["l_det"] / float(total_items) if total_items > 0 else 0.0
    result["l_heat"] = loss_sums["l_heat"] / float(total_forged_items) if total_forged_items > 0 else 0.0
    result["l_mask"] = loss_sums["l_mask"] / float(total_forged_items) if total_forged_items > 0 else 0.0
    result["l_edge"] = loss_sums["l_edge"] / float(total_forged_items) if total_forged_items > 0 else 0.0
    result["loss"] = (
        float(cfg["train"]["lambda_det"]) * result["l_det"]
        + float(cfg["train"]["lambda_heat"]) * result["l_heat"]
        + float(cfg["train"]["lambda_mask"]) * result["l_mask"]
        + float(cfg["train"].get("lambda_edge", 0.1)) * result["l_edge"]
    )
    return result


def write_summary_csv(results: List[Dict[str, Any]], path: Path) -> None:
    headers = ["指标"] + [result["dataset"] for result in results] + ["加权平均值"]
    rows = []

    total_samples = sum(result["num_samples"] for result in results)
    total_forged = sum(result["num_forged"] for result in results)

    metric_map = [
        ("Image-F1", "image_f1", "num_samples"),
        ("Image-ACC", "image_acc", "num_samples"),
        ("Pixel-F1", "pixel_f1", "num_forged"),
        ("Pixel-IoU", "pixel_iou", "num_forged"),
    ]
    for label, key, weight_key in metric_map:
        row = [label]
        weighted_sum = 0.0
        weighted_count = 0
        for result in results:
            row.append(f"{result[key]:.6f}")
            weighted_sum += float(result[key]) * int(result[weight_key])
            weighted_count += int(result[weight_key])
        row.append(f"{(weighted_sum / weighted_count) if weighted_count > 0 else 0.0:.6f}")
        rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def write_breakdown_csv(results: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Real-ACC", "Real-Count", "Forged-ACC", "Forged-Count"])
        for result in results:
            writer.writerow(
                [
                    result["dataset"],
                    f"{result['real_acc']:.6f}",
                    result["num_real"],
                    f"{result['forged_acc']:.6f}",
                    result["num_forged"],
                ]
            )


def build_magicbrush_dataset(cfg: Dict[str, Any], split: str):
    return MagicBrushDataset(
        manifest_path=cfg["data"]["manifests"][split],
        image_size=cfg["data"]["image_size"],
        processor_name_or_path=cfg["model"]["backbone"]["name_or_path"],
        edge_kernel_size=int(cfg["train"].get("edge_kernel_size", 5)),
    )


def _merge_eval_cfg(cli_cfg: Dict[str, Any], ckpt_cfg: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    cfg = copy.deepcopy(ckpt_cfg if ckpt_cfg else cli_cfg)
    if "data" not in cfg:
        cfg["data"] = {}
    if "model" not in cfg:
        cfg["model"] = {}
    if "train" not in cfg:
        cfg["train"] = {}
    if cli_cfg:
        cfg["output_dir"] = cli_cfg.get("output_dir", cfg.get("output_dir", str(run_dir)))
        if "data" in cli_cfg and "manifests" in cli_cfg["data"]:
            cfg["data"]["manifests"] = cli_cfg["data"]["manifests"]
    cfg["output_dir"] = str(run_dir)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--vis-path", type=str, default="")
    parser.add_argument("--max-vis-items", type=int, default=5)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--eval-all-datasets", action="store_true")
    parser.add_argument(
        "--autosplice-txt",
        type=str,
        default="/home/tione/notebook/users/youqiwang/Agent_All/Datasets/ori/test/AutoSplice_tp/AutoSplice_tp.txt",
    )
    parser.add_argument(
        "--autosplice-clean-dir",
        type=str,
        default="/home/tione/notebook/users/youqiwang/Agent_All/Datasets/ori/test/AutoSplice_au",
    )
    parser.add_argument(
        "--glide-txt",
        type=str,
        default="/home/tione/notebook/users/youqiwang/Agent_All/Datasets/ori/test/glide_tp/glide_tp.txt",
    )
    parser.add_argument(
        "--glide-clean-dir",
        type=str,
        default="/home/tione/notebook/users/youqiwang/Agent_All/Datasets/ori/test/glide_au",
    )
    parser.add_argument("--summary-csv", type=str, default="")
    parser.add_argument("--breakdown-csv", type=str, default="")
    args = parser.parse_args()

    requested_num_gpus = max(1, int(args.num_gpus))
    available_num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if requested_num_gpus > 1 and available_num_gpus < requested_num_gpus:
        raise ValueError(f"requested {requested_num_gpus} GPUs, but only {available_num_gpus} are available")
    device = torch.device("cuda:0" if available_num_gpus > 0 else "cpu")
    cli_cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    run_dir = Path(args.checkpoint).resolve().parent if args.checkpoint else resolve_eval_output_dir(cli_cfg["output_dir"])
    default_ckpt = run_dir / "best_by_iou.pt"
    if not default_ckpt.exists():
        default_ckpt = run_dir / "best_by_iou_full.pt"
    ckpt_path = args.checkpoint or str(default_ckpt)
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    cfg = _merge_eval_cfg(cli_cfg, ckpt.get("cfg") if isinstance(ckpt.get("cfg"), dict) else {}, run_dir)
    eval_batch_size = int(args.batch_size) if int(args.batch_size) > 0 else int(cfg["data"]["batch_size"])

    base_model = Stage1ForgeryModel(cfg["model"]).to(device)
    load_info = load_stage1_checkpoint_into_model(base_model, ckpt)
    print(
        json.dumps(
            {
                "checkpoint": ckpt_path,
                "using_checkpoint_cfg": bool(ckpt.get("cfg")),
                "load_missing_keys": load_info["missing"][:50],
                "load_unexpected_keys": load_info["unexpected"][:50],
                "num_missing_keys": len(load_info["missing"]),
                "num_unexpected_keys": len(load_info["unexpected"]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    model = EvalStage1Wrapper(base_model).to(device)
    if available_num_gpus > 1 and requested_num_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(requested_num_gpus)))
    model.eval()

    if args.eval_all_datasets:
        dataset_specs: List[Tuple[str, object]] = [
            ("MagicBrush-Test", build_magicbrush_dataset(cfg, "test")),
            (
                "CocoGlide",
                OODEvalDataset(
                    forged_txt_path=args.glide_txt,
                    clean_dir=args.glide_clean_dir,
                    image_size=cfg["data"]["image_size"],
                    processor_name_or_path=cfg["model"]["backbone"]["name_or_path"],
                    edge_kernel_size=int(cfg["train"].get("edge_kernel_size", 5)),
                    dataset_name="CocoGlide",
                ),
            ),
            (
                "AutoSplice",
                OODEvalDataset(
                    forged_txt_path=args.autosplice_txt,
                    clean_dir=args.autosplice_clean_dir,
                    image_size=cfg["data"]["image_size"],
                    processor_name_or_path=cfg["model"]["backbone"]["name_or_path"],
                    edge_kernel_size=int(cfg["train"].get("edge_kernel_size", 5)),
                    dataset_name="AutoSplice",
                ),
            ),
        ]

        results: List[Dict[str, Any]] = []
        for dataset_name, dataset in dataset_specs:
            loader = build_loader(dataset, cfg, batch_size=eval_batch_size)
            vis_path = str(run_dir / f"{dataset_name.replace('-', '_').replace(' ', '_')}_vis_annotated.png")
            result = evaluate_loader(
                dataset_name=dataset_name,
                loader=loader,
                model=model,
                device=device,
                num_gpus=requested_num_gpus if available_num_gpus > 0 else 0,
                cfg=cfg,
                vis_path=vis_path,
                max_vis_items=args.max_vis_items,
            )
            results.append(result)

        summary_csv = Path(args.summary_csv) if args.summary_csv else run_dir / "cross_dataset_summary.csv"
        breakdown_csv = Path(args.breakdown_csv) if args.breakdown_csv else run_dir / "cross_dataset_class_breakdown.csv"
        write_summary_csv(results, summary_csv)
        write_breakdown_csv(results, breakdown_csv)
        print(
            json.dumps(
                {
                    "checkpoint": ckpt_path,
                    "summary_csv": str(summary_csv),
                    "breakdown_csv": str(breakdown_csv),
                    "results": results,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    dataset = build_magicbrush_dataset(cfg, args.split)
    loader = build_loader(dataset, cfg, batch_size=eval_batch_size)
    vis_path = args.vis_path or str(run_dir / f"{args.split}_vis_annotated.png")
    result = evaluate_loader(
        dataset_name=f"MagicBrush-{args.split}",
        loader=loader,
        model=model,
        device=device,
        num_gpus=requested_num_gpus if available_num_gpus > 0 else 0,
        cfg=cfg,
        vis_path=vis_path,
        max_vis_items=args.max_vis_items,
    )
    summary_csv = Path(args.summary_csv) if args.summary_csv else run_dir / f"magicbrush_{args.split}_summary.csv"
    breakdown_csv = Path(args.breakdown_csv) if args.breakdown_csv else run_dir / f"magicbrush_{args.split}_class_breakdown.csv"
    write_summary_csv([result], summary_csv)
    write_breakdown_csv([result], breakdown_csv)
    print(
        json.dumps(
            {
                args.split: result,
                "summary_csv": str(summary_csv),
                "breakdown_csv": str(breakdown_csv),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
