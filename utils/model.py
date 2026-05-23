from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchcam.methods import GradCAMpp
from torchcam.utils import overlay_mask
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image


# Streamlit's source watcher may inspect torch.classes.__path__ and trigger
# a harmless but noisy PyTorch custom-class error. Make it a normal path-like
# value before Streamlit scans loaded modules.
try:
    torch.classes.__path__ = []
except Exception:
    pass


LABELS = ["N", "D", "G", "C", "A", "H", "M", "O"]
IMG_SIZE = (512, 512)
DISEASE_THRESHOLD_FLOOR = float(os.getenv("RETINASCOPE_DISEASE_THRESHOLD_FLOOR", "0.50"))
NORMAL_THRESHOLD_FLOOR = float(os.getenv("RETINASCOPE_NORMAL_THRESHOLD_FLOOR", "0.50"))
MAX_POSITIVE_DISEASES = int(os.getenv("RETINASCOPE_MAX_POSITIVE_DISEASES", "1"))
ENABLE_SCORECAM = os.getenv("RETINASCOPE_ENABLE_SCORECAM", "1").strip() == "1"
IMG_TF = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def calibrate(p_raw: float, th: float, temp: float = 2.0) -> float:
    # Treat the learned threshold as the logistic midpoint.
    z = (p_raw - th) * temp / (th * (1 - th) + 1e-6)
    return float(1 / (1 + np.exp(-z)))


def effective_threshold(label: str, learned_threshold: float) -> float:
    floor = NORMAL_THRESHOLD_FLOOR if label == "N" else DISEASE_THRESHOLD_FLOOR
    return max(float(learned_threshold), floor)


class ResNetBinary(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = models.convnext_base(
            weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        in_ch = 1024
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_ch, eps=1e-6),
            nn.Flatten(1),
            nn.Linear(in_ch, 1),
        )

    def forward(self, x):
        x = self.backbone(x)[:, :, 0, 0]
        return self.classifier(x)


@lru_cache(maxsize=32)
def _load_label_model(label: str, models_dir: str, device: str):
    ckpt_path = Path(models_dir) / f"best_{label}_fold5.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ResNetBinary(pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    threshold = float(ckpt.get("threshold", 0.5))
    return model, threshold


def _safe_gradcam(model, x, display_image):
    if not ENABLE_SCORECAM:
        return display_image.copy(), None
    try:
        cam_extractor = GradCAMpp(model, target_layer="backbone.0.7")
        with torch.enable_grad():
            out = model(x.clone().requires_grad_(True))
            activation_map = cam_extractor(class_idx=0, scores=out)
        cam_extractor.remove_hooks()
        return overlay_mask(
            display_image,
            to_pil_image(activation_map[0].squeeze(0), mode="F"),
            alpha=0.55,
        ), None
    except Exception as exc:
        return display_image.copy(), str(exc)


def predict(image: Image.Image, models_dir: Path, device="cpu"):
    """Return probs(8,), preds(8,), ScoreCAM images, and CAM error messages."""
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    device = str(device)
    models_dir = str(Path(models_dir).resolve())
    image = image.convert("RGB")
    x = IMG_TF(image).unsqueeze(0).to(device)
    display_image = image.resize(IMG_SIZE)
    probs, thresholds, cams, cam_errors = [], [], [], {}

    for label in LABELS:
        model, threshold = _load_label_model(label, models_dir, device)

        with torch.inference_mode():
            logit = model(x)
        result, cam_error = _safe_gradcam(model, x, display_image)
        cams.append(result)
        if cam_error:
            cam_errors[label] = cam_error

        p_raw = torch.sigmoid(logit).item()
        probs.append(float(p_raw))
        thresholds.append(effective_threshold(label, threshold))

    probs_np = np.array(probs, dtype=np.float32)
    thresholds_np = np.array(thresholds, dtype=np.float32)
    preds_np = (probs_np >= thresholds_np).astype(int)

    disease_indices = list(range(1, len(LABELS)))
    positive_diseases = [idx for idx in disease_indices if preds_np[idx] == 1]
    if MAX_POSITIVE_DISEASES > 0 and len(positive_diseases) > MAX_POSITIVE_DISEASES:
        keep = sorted(positive_diseases, key=lambda idx: probs_np[idx], reverse=True)[:MAX_POSITIVE_DISEASES]
        for idx in positive_diseases:
            preds_np[idx] = 1 if idx in keep else 0

    if preds_np[1:].any():
        preds_np[0] = 0

    return probs_np, preds_np, cams, cam_errors
