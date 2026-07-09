from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.model import LABELS, generate_cam, predict_scores


def merge_pair(left_path: Path, right_path: Path) -> Image.Image:
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    height = max(left.height, right.height)
    merged = Image.new("RGB", (left.width + right.width, height), "black")
    merged.paste(left, (0, 0))
    merged.paste(right, (left.width, 0))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local RetinaScope inference smoke test.")
    parser.add_argument("--left", type=Path, default=ROOT / "test" / "images" / "0_left.jpg")
    parser.add_argument("--right", type=Path, default=ROOT / "test" / "images" / "0_right.jpg")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "artifacts" / "models")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    image = merge_pair(args.left, args.right)
    prediction = predict_scores(image, args.models_dir, device=args.device)
    top_index = int(prediction.probs.argmax())
    top_label = LABELS[top_index]
    cam = generate_cam(image, top_label, args.models_dir, device=args.device)
    print(
        json.dumps(
            {
                "device": prediction.device,
                "model_version": prediction.model_version,
                "tta_enabled": prediction.tta_enabled,
                "top_label": top_label,
                "top_probability": round(float(prediction.probs[top_index]), 6),
                "predictions": prediction.preds.tolist(),
                "inference_ms": prediction.inference_ms,
                "cam_ok": cam.image is not None,
                "cam_ms": cam.generation_ms,
                "cam_error": cam.error,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
