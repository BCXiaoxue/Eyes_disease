from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.paths import MODELS_DIR as DEFAULT_MODELS_DIR
from utils.paths import PREDICTED_CSV, TEST_IMAGES_DIR, VALIDATION_CSV


LABELS = ["N", "D", "G", "C", "A", "H", "M", "O"]


parser = argparse.ArgumentParser("Test Prediction Script")
parser.add_argument("--csv", default=str(VALIDATION_CSV), help="Prediction CSV path")
parser.add_argument("--img_dir", default=str(TEST_IMAGES_DIR / "merged"), help="Directory containing merged images")
parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR), help="Directory containing model checkpoints")
parser.add_argument("--batch", type=int, default=32, help="Batch size")
args = parser.parse_args()

CSV_PATH = args.csv
IMG_DIR = Path(args.img_dir)
MODELS_DIR = Path(args.models_dir)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageOnlyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = str(self.df.iloc[idx, 0])
        img_path = self.img_dir / f"{img_id}_merge.jpg"
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class ResNetBinary(nn.Module):
    def __init__(self, pretrained=True):
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


val_tf = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

df = pd.read_csv(CSV_PATH)
dataset = ImageOnlyDataset(df, IMG_DIR, val_tf)
loader = DataLoader(dataset, batch_size=args.batch, shuffle=False)

all_probs = np.zeros((len(dataset), len(LABELS)), dtype=np.float32)
thresholds = []

for idx, cls in enumerate(LABELS):
    ckpt_path = MODELS_DIR / f"best_{cls}_fold5.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    th = ckpt.get("threshold", 0.5)
    thresholds.append(th)

    model = ResNetBinary().to(DEVICE)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    probs_cls = []
    with torch.no_grad():
        for imgs in tqdm(loader, desc=f"Infer {cls}"):
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs_cls.append(torch.sigmoid(logits).cpu().numpy())
    all_probs[:, idx] = np.concatenate(probs_cls).ravel()

preds = np.zeros_like(all_probs, dtype=int)
for i, th in enumerate(thresholds):
    preds[:, i] = (all_probs[:, i] >= th).astype(int)

for i, col in enumerate(LABELS):
    df[col] = preds[:, i]

df.to_csv(PREDICTED_CSV, index=False)
print(f"Prediction results saved to {PREDICTED_CSV}")
