from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.binocular_label_graph import (
    LABELS,
    ODIRExperimentDataset,
    build_transforms,
    load_experiment_checkpoint,
)
from utils.paths import PREDICTED_CSV, TEST_IMAGES_DIR, VALIDATION_CSV


def unpack_batch(batch, input_mode: str, device: torch.device):
    if input_mode == "merged":
        images, _ = batch
        return (images.to(device),)
    left, right, _ = batch
    return left.to(device), right.to(device)


def main() -> None:
    parser = argparse.ArgumentParser("Predict with a RetinaScope experiment checkpoint.")
    parser.add_argument("--csv", default=str(VALIDATION_CSV))
    parser.add_argument("--image_root", default=str(TEST_IMAGES_DIR))
    parser.add_argument("--merged_dir", default=str(TEST_IMAGES_DIR / "merged"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=str(PREDICTED_CSV))
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, thresholds, ckpt = load_experiment_checkpoint(args.checkpoint, device=device)
    config = ckpt["config"]
    df = pd.read_csv(args.csv)
    predict_df = df.copy()
    for label in LABELS:
        if label not in predict_df.columns:
            predict_df[label] = 0
    dataset = ODIRExperimentDataset(
        predict_df,
        image_root=args.image_root,
        merged_dir=args.merged_dir,
        input_mode=config["input_mode"],
        transform=build_transforms(False, config.get("img_size", 512)),
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    probs_all = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict"):
            logits = model(*unpack_batch(batch, config["input_mode"], device))
            probs_all.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(probs_all)
    preds = (probs >= thresholds.reshape(1, -1)).astype(int)
    for idx, label in enumerate(LABELS):
        df[label] = preds[:, idx]
        df[f"{label}_prob"] = probs[:, idx]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Prediction results saved to {output}")


if __name__ == "__main__":
    main()
