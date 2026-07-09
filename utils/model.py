from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from utils.binocular_label_graph import load_experiment_checkpoint


try:
    torch.classes.__path__ = []
except Exception:
    pass


LABELS = ["N", "D", "G", "C", "A", "H", "M", "O"]
IMG_SIZE = (512, 512)
MODEL_FILENAME = os.getenv("RETINASCOPE_MODEL_FILE", "best_swin_tiny_linear_asl.pth")
ENABLE_TTA = os.getenv("RETINASCOPE_ENABLE_TTA", "1").strip() == "1"
TTA_EVAL_FILENAME = os.getenv("RETINASCOPE_TTA_EVAL_FILE", "eval_swin_tiny_linear_asl_tta.json")
IMG_TF = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


@dataclass(frozen=True)
class PredictionResult:
    probs: np.ndarray
    preds: np.ndarray
    thresholds: np.ndarray
    model_version: str
    device: str
    tta_enabled: bool
    inference_ms: float


@dataclass(frozen=True)
class CamResult:
    label: str
    image: Image.Image | None
    error: str
    generation_ms: float


def _resolve_model_path(models_dir: str | Path) -> Path:
    models_dir = Path(models_dir)
    preferred = models_dir / MODEL_FILENAME
    if preferred.exists():
        return preferred
    candidates = sorted(models_dir.glob("best_swin*_asl.pth"))
    return candidates[0] if candidates else preferred


def _resolve_tta_eval_path(models_dir: str | Path) -> Path:
    return Path(models_dir) / TTA_EVAL_FILENAME


def _file_fingerprint(path: Path) -> str:
    if not path.exists():
        return "missing"
    stat = path.stat()
    raw = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def get_model_fingerprint(models_dir: str | Path) -> str:
    model_path = _resolve_model_path(models_dir)
    eval_path = _resolve_tta_eval_path(models_dir)
    raw = f"{_file_fingerprint(model_path)}:{_file_fingerprint(eval_path)}:tta={int(ENABLE_TTA)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=4)
def _load_multilabel_model(models_dir: str, device: str, fingerprint: str):
    del fingerprint
    ckpt_path = _resolve_model_path(models_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    return load_experiment_checkpoint(ckpt_path, device=device)


@lru_cache(maxsize=8)
def _load_threshold_override(models_dir: str, fingerprint: str) -> np.ndarray | None:
    del fingerprint
    eval_path = _resolve_tta_eval_path(models_dir)
    if not eval_path.exists():
        return None
    try:
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        thresholds = data["calibrated_threshold_metrics"]["thresholds"]
        return np.array([float(thresholds[label]) for label in LABELS], dtype=np.float32)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _normalise_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return str(device)


def _predict_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    if ENABLE_TTA:
        logits = (logits + model(torch.flip(x, dims=[3]))) / 2.0
    return logits


def _apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if thresholds.shape != (len(LABELS),):
        thresholds = np.full(len(LABELS), 0.5, dtype=np.float32)
    preds = (np.asarray(probs, dtype=np.float32) >= thresholds).astype(int)
    if preds[1:].any():
        preds[0] = 0
    elif preds[0] == 0:
        preds[0] = 1
    return preds


def predict_scores(image: Image.Image, models_dir: str | Path, device: str = "cpu") -> PredictionResult:
    device = _normalise_device(device)
    models_dir = str(Path(models_dir).resolve())
    fingerprint = get_model_fingerprint(models_dir)
    image = image.convert("RGB")
    x = IMG_TF(image).unsqueeze(0).to(device)
    model, checkpoint_thresholds, _metadata = _load_multilabel_model(models_dir, device, fingerprint)

    started = time.perf_counter()
    with torch.inference_mode():
        logits = _predict_logits(model, x)
    inference_ms = (time.perf_counter() - started) * 1000.0

    probs = torch.sigmoid(logits).detach().cpu().numpy()[0].astype(np.float32)
    thresholds = _load_threshold_override(models_dir, fingerprint)
    if thresholds is None:
        thresholds = np.asarray(checkpoint_thresholds, dtype=np.float32)
    if thresholds.shape != (len(LABELS),):
        thresholds = np.full(len(LABELS), 0.5, dtype=np.float32)
    preds = _apply_thresholds(probs, thresholds)
    return PredictionResult(
        probs=probs,
        preds=preds,
        thresholds=thresholds,
        model_version=fingerprint,
        device=device,
        tta_enabled=ENABLE_TTA,
        inference_ms=round(inference_ms, 1),
    )


def _heatmap_overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    cam = np.clip(cam, 0.0, 1.0)
    heat = np.zeros((*cam.shape, 3), dtype=np.uint8)
    heat[..., 0] = np.clip(255 * np.minimum(1.0, cam * 2.0), 0, 255).astype(np.uint8)
    heat[..., 1] = np.clip(255 * np.maximum(0.0, (cam - 0.35) * 1.6), 0, 255).astype(np.uint8)
    heat[..., 2] = np.clip(120 * np.maximum(0.0, 1.0 - cam * 2.5), 0, 255).astype(np.uint8)
    heat_img = Image.fromarray(heat, mode="RGB").resize(image.size, Image.BICUBIC)
    return Image.blend(image.convert("RGB"), heat_img, alpha=alpha)


def generate_cam(
    image: Image.Image,
    label: str,
    models_dir: str | Path,
    device: str = "cpu",
) -> CamResult:
    if label not in LABELS:
        return CamResult(label=label, image=None, error=f"Unknown label: {label}", generation_ms=0.0)

    device = _normalise_device(device)
    models_dir = str(Path(models_dir).resolve())
    fingerprint = get_model_fingerprint(models_dir)
    source_image = image.convert("RGB")
    display_image = source_image.resize(IMG_SIZE)
    x = IMG_TF(source_image).unsqueeze(0).to(device)
    model, _thresholds, _metadata = _load_multilabel_model(models_dir, device, fingerprint)
    handle = None
    started = time.perf_counter()
    try:
        target_layer = dict(model.named_modules()).get("backbone.norm")
        if target_layer is None:
            raise RuntimeError("target layer backbone.norm not found")
        activations = []

        def save_activation(_module, _inputs, output):
            output.retain_grad()
            activations.append(output)

        handle = target_layer.register_forward_hook(save_activation)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(x.detach().clone().requires_grad_(True))
            if not activations:
                raise RuntimeError("target activation was not captured")
            activation = activations[-1]
            logits[0, LABELS.index(label)].backward()
            grad = activation.grad
            if grad is None:
                raise RuntimeError("target gradient was not captured")
            act = activation.detach()
            if act.ndim == 4 and act.shape[-1] >= act.shape[1]:
                weights = grad.mean(dim=(1, 2), keepdim=True)
                cam = torch.relu((act * weights).sum(dim=-1))[0]
            elif act.ndim == 4:
                weights = grad.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((act * weights).sum(dim=1))[0]
            else:
                raise RuntimeError(f"unsupported activation shape: {tuple(act.shape)}")
            cam = cam - cam.min()
            cam_np = (cam / cam.max().clamp(min=1e-6)).detach().cpu().numpy()
            output = _heatmap_overlay(display_image, cam_np)
        elapsed = (time.perf_counter() - started) * 1000.0
        return CamResult(label=label, image=output, error="", generation_ms=round(elapsed, 1))
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000.0
        return CamResult(label=label, image=None, error=f"Swin Grad-CAM 生成失败：{exc}", generation_ms=round(elapsed, 1))
    finally:
        if handle is not None:
            handle.remove()
