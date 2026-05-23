from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.paths import VAL_CSV, TRAIN_IMAGES_DIR, MODELS_DIR as DEFAULT_MODELS_DIR, FIGS_DIR

LABELS = ["N", "D", "G", "C", "A", "H", "M", "O"]

# --------------------------------------------------
# 1. Args
# --------------------------------------------------
parser = argparse.ArgumentParser("Multi‑label evaluation of 8 models")
parser.add_argument("--csv", default=str(VAL_CSV))
parser.add_argument("--img_dir", default=str(TRAIN_IMAGES_DIR / "merged"))
parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR))
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--show", action="store_true", help="show confusion matrices instead of saving")
args = parser.parse_args()

CSV_PATH   = args.csv
IMG_DIR    = Path(args.img_dir)
MODELS_DIR = Path(args.models_dir)
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# 2. Dataset
# --------------------------------------------------
class ImageOnlyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = str(self.df.iloc[idx, 0])
        img = Image.open(self.img_dir / f"{img_id}_merge.jpg").convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

# --------------------------------------------------
# 3. Model (ConvNeXt backbone + LN + Linear)
# --------------------------------------------------
class ResNetBinary(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None)
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

# --------------------------------------------------
# 4. Data loading
# --------------------------------------------------
df = pd.read_csv(CSV_PATH)
true_labels = df[LABELS].values.astype(int)

val_tf = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

dataset = ImageOnlyDataset(df, IMG_DIR, val_tf)
loader  = DataLoader(dataset, batch_size=args.batch, shuffle=False)

# --------------------------------------------------
# 5. Inference
# --------------------------------------------------
all_probs = np.zeros((len(dataset), len(LABELS)), dtype=np.float32)
thresholds = []

for idx, cls in enumerate(LABELS):
    ckpt_path = MODELS_DIR / f"best_{cls}_fold5.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")
    ckpt = torch.load(ckpt_path, map_location=DEVICE,weights_only=False)
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

# --------------------------------------------------
# 6. Thresholding & global metrics
# --------------------------------------------------
preds = np.zeros_like(all_probs, dtype=int)
for i, th in enumerate(thresholds):
    preds[:, i] = (all_probs[:, i] >= th).astype(int)

flat_true, flat_pred = true_labels.ravel(), preds.ravel()
acc_global  = accuracy_score(flat_true, flat_pred)
prec_global = precision_score(flat_true, flat_pred, average="weighted", zero_division=0)
rec_global  = recall_score(flat_true, flat_pred, average="weighted", zero_division=0)

print(f"\n===== Overall metrics on {CSV_PATH} =====")
print(f"Accuracy  : {acc_global:.4f}")
print(f"Precision : {prec_global:.4f}")
print(f"Recall    : {rec_global:.4f}\n")

# --------------------------------------------------
# 7. Per‑label metrics & confusion matrices
# --------------------------------------------------
per_acc  = ((preds == true_labels).sum(0) / len(dataset)).astype(float)
per_prec = precision_score(true_labels, preds, average=None, zero_division=0)
per_rec  = recall_score(true_labels, preds, average=None, zero_division=0)
per_f1   = f1_score(true_labels, preds, average=None, zero_division=0)
per_auc  = [roc_auc_score(true_labels[:, i], all_probs[:, i]) for i in range(len(LABELS))]

print("Per‑label metrics:")
print("Label  Acc   Prec  Rec   F1    AUC   Th")
for i, cls in enumerate(LABELS):
    print(f" {cls:>2}  {per_acc[i]:.3f}  {per_prec[i]:.3f}  {per_rec[i]:.3f}  {per_f1[i]:.3f}  {per_auc[i]:.3f}  {thresholds[i]:.2f}")

# -------- Confusion matrix plots --------
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

for i, cls in enumerate(LABELS):
    cm = confusion_matrix(true_labels[:, i], preds[:, i])
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap="Blues")

    ax.set_title(f"Confusion Matrix – {cls}")

    # Configure axis labels by class.
    if cls == "N":
        x_labels = ["Normal", "Disease"]
        y_labels = ["Normal", "Disease"]
    else:
        x_labels = ["No Disease", "Disease"]
        y_labels = ["No Disease", "Disease"]

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Annotate each cell.
    for (j, k), val in np.ndenumerate(cm):
        ax.text(k, j, str(val), ha='center', va='center',
                color="white" if cm[j, k] > cm.max() / 2 else "black")

    plt.tight_layout()

    if args.show:
        plt.show()
    else:
        FIGS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGS_DIR / f"confmat_{cls}.png")
    
    plt.close(fig)

print("\nConfusion matrices saved as confmat_<label>.png (use --show to display).")
