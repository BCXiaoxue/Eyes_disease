from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.binocular_label_graph import LABELS, ODIRExperimentDataset, build_transforms, load_experiment_checkpoint
from utils.paths import TRAIN_IMAGES_DIR, VAL_CSV


SERVER_ROOT = Path("/root/wangchen/tiansukai")


def ensure_under_server_root(paths: list[Path]) -> None:
    for path in paths:
        resolved = path.resolve()
        try:
            resolved.relative_to(SERVER_ROOT)
        except ValueError as exc:
            raise ValueError(f"Server evaluation path must stay under {SERVER_ROOT}: {resolved}") from exc


def unpack_batch(batch, input_mode: str, device: torch.device):
    if input_mode == "merged":
        images, targets = batch
        return (images.to(device, non_blocking=True),), targets.to(device, non_blocking=True)
    left, right, targets = batch
    return (left.to(device, non_blocking=True), right.to(device, non_blocking=True)), targets.to(device, non_blocking=True)


def best_thresholds(targets: np.ndarray, probs: np.ndarray) -> np.ndarray:
    thresholds = []
    for idx in range(targets.shape[1]):
        if len(np.unique(targets[:, idx])) < 2:
            thresholds.append(0.5)
            continue
        precision, recall, raw_thresholds = precision_recall_curve(targets[:, idx], probs[:, idx])
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-8, None)
        best_idx = int(np.nanargmax(f1))
        thresholds.append(float(raw_thresholds[min(best_idx, len(raw_thresholds) - 1)]) if len(raw_thresholds) else 0.5)
    return np.array(thresholds, dtype=np.float32)


def metrics_for(targets: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> dict:
    preds = (probs >= thresholds.reshape(1, -1)).astype(int)
    per_auc = []
    for idx in range(targets.shape[1]):
        if len(np.unique(targets[:, idx])) < 2:
            per_auc.append(float("nan"))
        else:
            per_auc.append(float(roc_auc_score(targets[:, idx], probs[:, idx])))
    return {
        "macro_auc": float(np.nanmean(per_auc)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(targets, preds, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(targets, preds, average="weighted", zero_division=0)),
        "per_label_auc": {label: float(value) for label, value in zip(LABELS, per_auc)},
        "per_label_f1": {
            label: float(value)
            for label, value in zip(LABELS, f1_score(targets, preds, average=None, zero_division=0))
        },
        "thresholds": {label: float(value) for label, value in zip(LABELS, thresholds)},
    }


def evaluate(model, loader, device: torch.device, input_mode: str, tta: bool) -> tuple[np.ndarray, np.ndarray]:
    probs_all, targets_all = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            inputs, targets = unpack_batch(batch, input_mode, device)
            logits = model(*inputs)
            if tta:
                flipped_inputs = tuple(torch.flip(input_tensor, dims=[3]) for input_tensor in inputs)
                logits = (logits + model(*flipped_inputs)) / 2.0
            probs_all.append(torch.sigmoid(logits).cpu())
            targets_all.append(targets.cpu())
    return torch.cat(probs_all).numpy(), torch.cat(targets_all).numpy().astype(int)


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate a RetinaScope experiment checkpoint with optional TTA.")
    parser.add_argument("--val_csv", default=str(VAL_CSV))
    parser.add_argument("--image_root", default=str(TRAIN_IMAGES_DIR))
    parser.add_argument("--merged_dir", default=str(TRAIN_IMAGES_DIR / "merged"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--enforce_server_root", action="store_true")
    args = parser.parse_args()

    paths = [Path(args.val_csv), Path(args.image_root), Path(args.merged_dir), Path(args.checkpoint), Path(args.output).parent]
    if args.enforce_server_root:
        ensure_under_server_root(paths)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, learned_thresholds, ckpt = load_experiment_checkpoint(args.checkpoint, device=device)
    config = ckpt["config"]
    df = pd.read_csv(args.val_csv)
    dataset = ODIRExperimentDataset(
        df,
        image_root=args.image_root,
        merged_dir=args.merged_dir,
        input_mode=config["input_mode"],
        transform=build_transforms(False, config.get("img_size", 512)),
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    probs, targets = evaluate(model, loader, device, config["input_mode"], args.tta)
    calibrated_thresholds = best_thresholds(targets, probs)
    result = {
        "checkpoint": args.checkpoint,
        "config": config,
        "tta": args.tta,
        "learned_threshold_metrics": metrics_for(targets, probs, learned_thresholds),
        "calibrated_threshold_metrics": metrics_for(targets, probs, calibrated_thresholds),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    best = result["calibrated_threshold_metrics"]
    print(
        f"calibrated macro_auc={best['macro_auc']:.4f} "
        f"macro_f1={best['macro_f1']:.4f} output={output}"
    )


if __name__ == "__main__":
    main()
