from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file as load_safetensors
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.binocular_label_graph import (
    LABELS,
    PRESETS,
    AsymmetricLoss,
    DistributionBalancedLoss,
    FocalBCEWithLogitsLoss,
    ODIRExperimentDataset,
    WeightedAsymmetricLoss,
    build_label_graph,
    build_model,
    build_transforms,
    compute_distribution_balanced_weights,
    compute_effective_class_weight,
    compute_pos_weight,
)
from utils.paths import NEW_MODELS_DIR, TRAIN_CSV, TRAIN_IMAGES_DIR, VAL_CSV


SERVER_ROOT = Path("/root/wangchen/tiansukai")


def parse_args():
    parser = argparse.ArgumentParser("Train RetinaScope baselines and binocular Label-GCN model.")
    parser.add_argument("--train_csv", default=str(TRAIN_CSV))
    parser.add_argument("--val_csv", default=str(VAL_CSV))
    parser.add_argument("--image_root", default=str(TRAIN_IMAGES_DIR))
    parser.add_argument("--merged_dir", default=str(TRAIN_IMAGES_DIR / "merged"))
    parser.add_argument("--output_dir", default=str(NEW_MODELS_DIR / "binocular_label_graph"))
    parser.add_argument("--preset", choices=sorted(PRESETS), default="dual_branch_label_gcn_asl")
    parser.add_argument("--loss", choices=["bce", "asl", "focal", "db", "asl_cb"], default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--asl_gamma_neg", type=float, default=4.0)
    parser.add_argument("--asl_gamma_pos", type=float, default=1.0)
    parser.add_argument("--asl_clip", type=float, default=0.05)
    parser.add_argument("--cb_beta", type=float, default=0.9999)
    parser.add_argument("--mixup_alpha", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--no_sampler", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--pretrained_checkpoint", default=None)
    parser.add_argument("--tta_eval", action="store_true")
    parser.add_argument("--enforce_server_root", action="store_true")
    return parser.parse_args()


def ensure_under_server_root(paths: list[Path]) -> None:
    for path in paths:
        resolved = path.resolve()
        try:
            resolved.relative_to(SERVER_ROOT)
        except ValueError as exc:
            raise ValueError(f"Server training path must stay under {SERVER_ROOT}: {resolved}") from exc


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_sampler(df: pd.DataFrame):
    label_counts = df[LABELS].sum(axis=0).to_numpy(dtype=np.float32)
    inv_freq = 1.0 / np.clip(label_counts, 1.0, None)
    weights = df[LABELS].to_numpy(dtype=np.float32).dot(inv_freq)
    floor = weights[weights > 0].mean() if (weights > 0).any() else 1.0
    weights = np.where(weights > 0, weights, floor)
    return WeightedRandomSampler(weights.tolist(), num_samples=len(weights), replacement=True)


def unpack_batch(batch, input_mode: str, device: torch.device):
    if input_mode == "merged":
        images, targets = batch
        return (images.to(device, non_blocking=True),), targets.to(device, non_blocking=True)
    left, right, targets = batch
    return (left.to(device, non_blocking=True), right.to(device, non_blocking=True)), targets.to(device, non_blocking=True)


def autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda")
    return torch.cuda.amp.autocast()


def build_grad_scaler(device: torch.device):
    if device.type != "cuda":
        return torch.cuda.amp.GradScaler(enabled=False)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


def apply_mixup(inputs: tuple[torch.Tensor, ...], targets: torch.Tensor, alpha: float):
    if alpha <= 0 or targets.size(0) < 2:
        return inputs, targets
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(targets.size(0), device=targets.device)
    mixed_inputs = tuple(lam * input_tensor + (1.0 - lam) * input_tensor[index] for input_tensor in inputs)
    mixed_targets = lam * targets + (1.0 - lam) * targets[index]
    return mixed_inputs, mixed_targets


def run_epoch(model, loader, criterion, optimizer, scaler, device: torch.device, input_mode: str, train: bool, tta: bool = False):
    model.train(train)
    losses, logits_all, targets_all = [], [], []
    for batch in tqdm(loader, desc="Train" if train else "Val", leave=False):
        inputs, targets = unpack_batch(batch, input_mode, device)
        if train:
            inputs, targets = apply_mixup(inputs, targets, getattr(criterion, "mixup_alpha", 0.0))
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with autocast_context(device):
                logits = model(*inputs)
                if tta and not train:
                    flipped_inputs = tuple(torch.flip(input_tensor, dims=[3]) for input_tensor in inputs)
                    logits = (logits + model(*flipped_inputs)) / 2.0
                loss = criterion(logits, targets)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        losses.append(float(loss.detach().cpu().item()) * targets.size(0))
        logits_all.append(logits.detach().cpu())
        targets_all.append(targets.detach().cpu())
    logits_tensor = torch.cat(logits_all).float()
    logits_tensor = torch.nan_to_num(logits_tensor, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
    targets_np = torch.cat(targets_all).numpy().astype(int)
    probs_np = torch.sigmoid(logits_tensor).numpy()
    return sum(losses) / len(loader.dataset), probs_np, targets_np


def best_thresholds(targets: np.ndarray, probs: np.ndarray) -> np.ndarray:
    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
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
    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
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


def build_loss(loss_name: str, train_df: pd.DataFrame, device: torch.device, args):
    if loss_name == "asl":
        return AsymmetricLoss(gamma_neg=args.asl_gamma_neg, gamma_pos=args.asl_gamma_pos, clip=args.asl_clip)
    if loss_name == "asl_cb":
        class_weight = compute_effective_class_weight(train_df, beta=args.cb_beta).to(device)
        return WeightedAsymmetricLoss(
            class_weight=class_weight,
            gamma_neg=args.asl_gamma_neg,
            gamma_pos=args.asl_gamma_pos,
            clip=args.asl_clip,
        )
    if loss_name == "db":
        pos_weight, neg_weight = compute_distribution_balanced_weights(train_df, beta=args.cb_beta)
        return DistributionBalancedLoss(
            pos_weight=pos_weight.to(device),
            neg_weight=neg_weight.to(device),
            gamma_neg=args.asl_gamma_neg,
            gamma_pos=args.asl_gamma_pos,
            clip=args.asl_clip,
        )
    pos_weight = compute_pos_weight(train_df).to(device)
    if loss_name == "focal":
        return FocalBCEWithLogitsLoss(pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def load_pretrained_checkpoint(model: nn.Module, checkpoint_path: str) -> tuple[int, int]:
    if checkpoint_path.endswith(".safetensors"):
        state_dict = load_safetensors(checkpoint_path, device="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_state = model.state_dict()
    compatible = {}
    for key, value in state_dict.items():
        clean_key = key.removeprefix("module.").removeprefix("model.")
        candidates = [clean_key, f"backbone.{clean_key}"]
        for candidate in candidates:
            if candidate in model_state and tuple(model_state[candidate].shape) == tuple(value.shape):
                compatible[candidate] = value
                break
    model.load_state_dict(compatible, strict=False)
    return len(compatible), len(state_dict)


def main() -> None:
    args = parse_args()
    if args.enforce_server_root:
        ensure_under_server_root(
            [Path(args.train_csv), Path(args.val_csv), Path(args.image_root), Path(args.merged_dir), Path(args.output_dir)]
        )

    set_seed(args.seed)
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    config = PRESETS[args.preset]
    config = type(config)(**{**config.to_dict(), "img_size": args.img_size})
    loss_name = args.loss or ("asl" if config.preset == "dual_branch_label_gcn_asl" else "bce")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_graph = build_label_graph(train_df)

    train_ds = ODIRExperimentDataset(
        train_df,
        image_root=args.image_root,
        merged_dir=args.merged_dir,
        input_mode=config.input_mode,
        transform=build_transforms(True, args.img_size),
    )
    val_ds = ODIRExperimentDataset(
        val_df,
        image_root=args.image_root,
        merged_dir=args.merged_dir,
        input_mode=config.input_mode,
        transform=build_transforms(False, args.img_size),
    )
    sampler = None if args.no_sampler else make_sampler(train_df)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        config,
        label_graph=label_graph,
        pretrained=(not args.no_pretrained and not args.pretrained_checkpoint),
    )
    if args.pretrained_checkpoint:
        loaded_keys, total_keys = load_pretrained_checkpoint(model, args.pretrained_checkpoint)
        print(f"Loaded pretrained checkpoint: {args.pretrained_checkpoint} ({loaded_keys}/{total_keys} keys)")
    model = model.to(device)
    criterion = build_loss(loss_name, train_df, device, args)
    setattr(criterion, "mixup_alpha", args.mixup_alpha)
    effective_lr = args.lr
    if config.preset == "dual_branch_label_gcn_asl" and args.lr == 1e-4:
        effective_lr = 3e-5
    optimizer = optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    scaler = build_grad_scaler(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history, best_record = [], {}
    best_score = -1.0
    best_path = output_dir / f"best_{config.preset}_{loss_name}.pth"

    print(f"Device: {device}")
    print(f"Preset: {config.preset} input_mode={config.input_mode} head={config.head} loss={loss_name} lr={effective_lr}")
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, scaler, device, config.input_mode, True)
        val_loss, val_probs, val_targets = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            scaler,
            device,
            config.input_mode,
            False,
            tta=args.tta_eval,
        )
        scheduler.step()
        thresholds = best_thresholds(val_targets, val_probs)
        metric_values = metrics_for(val_targets, val_probs, thresholds)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **metric_values,
        }
        history.append(record)
        score = metric_values["macro_auc"] + metric_values["macro_f1"]
        print(
            f"Ep{epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"macro_auc={metric_values['macro_auc']:.4f} macro_f1={metric_values['macro_f1']:.4f}"
        )
        if score > best_score:
            best_score = score
            best_record = record
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "labels": LABELS,
                    "thresholds": thresholds.tolist(),
                    "config": config.to_dict(),
                    "loss": loss_name,
                    "loss_params": {
                        "asl_gamma_neg": args.asl_gamma_neg,
                        "asl_gamma_pos": args.asl_gamma_pos,
                        "asl_clip": args.asl_clip,
                        "cb_beta": args.cb_beta,
                        "mixup_alpha": args.mixup_alpha,
                    },
                    "label_graph": label_graph.numpy().tolist(),
                    "metrics": best_record,
                    "history": history,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "tta_eval": args.tta_eval,
                },
                best_path,
            )

    report_path = output_dir / f"report_{config.preset}_{loss_name}.json"
    report_path.write_text(
        json.dumps(
            {
                "best": best_record,
                "history": history,
                "config": config.to_dict(),
                "loss": loss_name,
                "loss_params": {
                    "asl_gamma_neg": args.asl_gamma_neg,
                    "asl_gamma_pos": args.asl_gamma_pos,
                    "asl_clip": args.asl_clip,
                    "cb_beta": args.cb_beta,
                    "mixup_alpha": args.mixup_alpha,
                },
                "tta_eval": args.tta_eval,
                "pretrained": not args.no_pretrained,
                "pretrained_checkpoint": args.pretrained_checkpoint,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Best checkpoint: {best_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
