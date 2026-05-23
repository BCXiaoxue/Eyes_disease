import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.paths import TRAIN_CSV, TRAIN_IMAGES_DIR, NEW_MODELS_DIR

# args
parser = argparse.ArgumentParser(description="Binary CV for one label")
parser.add_argument("--csv",      default=str(TRAIN_CSV), help="CSV file path")
parser.add_argument("--img_dir",  default=str(TRAIN_IMAGES_DIR / "merged"), help="image directory")
parser.add_argument("--label",    default="N", choices=list("NDGCAHMO"), help="target label")
parser.add_argument("--folds",    type=int, default=5)
parser.add_argument("--epochs",   type=int, default=20)
parser.add_argument("--batch",    type=int, default=32)
parser.add_argument("--lr",       type=float, default=1e-4)
parser.add_argument("--lr_min",   type=float, default=1e-6)
parser.add_argument("--seed",     type=int, default=42)
args = parser.parse_args()

# configs
torch.manual_seed(args.seed)
np.random.seed(args.seed)

CSV_PATH = args.csv
IMG_DIR  = Path(args.img_dir)
LABEL    = args.label
N_FOLDS  = args.folds
EPOCHS   = args.epochs
LR_INIT  = args.lr
LR_MIN   = args.lr_min
BATCH    = args.batch
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training binary model for label '{LABEL}' on device {DEVICE}\n")

# load dataset
class SingleLabelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, label_col: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row["ID"])
        img = Image.open(self.img_dir / f"{img_id}_merge.jpg").convert("RGB")
        label = np.float32(row[self.label_col])
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label)

# image agumentation
train_tf = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# pretrained ConvNeXt backbone (custom classifier). ref:https://arxiv.org/pdf/2201.03545
class ResNetBinary(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None)
        # base = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        for param in self.backbone.parameters():
            param.requires_grad=False
        in_ch = 1024
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_ch, eps=1e-6),
            nn.Flatten(1),
            nn.Linear(in_ch, 1),
        )

    def forward(self, x):
        x = self.backbone(x)[:,:,0,0]
        return self.classifier(x)

# calculate metrics
def calc_metrics(labels, probs, thresh=0.5):
    preds = (probs >= thresh).astype(int)
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    auc  = roc_auc_score(labels, probs)
    return acc, prec, rec, auc


# train set tuning
def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    run_loss, all_labels, all_probs = 0.0, [], []
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        run_loss += loss.item() * imgs.size(0)
        all_labels.append(labels.cpu().numpy().ravel())
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy().ravel())
    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)
    acc, prec, rec, auc = calc_metrics(all_labels, all_probs)
    return run_loss / len(loader.dataset), acc, prec, rec, auc

# val set inference
def eval_epoch(model, loader, criterion):
    model.eval()
    run_loss, all_labels, all_probs = 0.0, [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            logits = model(imgs)
            loss = criterion(logits, labels)
            run_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            all_labels.append(labels.cpu().numpy().ravel())
            all_probs.append(probs)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    p, r, th = precision_recall_curve(all_labels, all_probs)
    f1 = 2 * p * r / (p + r + 1e-8)
    best_idx = f1.argmax() 
    best_th = th[max(best_idx - 1, 0)] if best_idx == len(th) else th[best_idx] # find best soft thresholding
    acc, prec, rec, auc = calc_metrics(all_labels, all_probs, best_th)
    return run_loss / len(loader.dataset), acc, prec, rec, auc, best_th


## 5-fold cross-valiadation
df = pd.read_csv(CSV_PATH)
if LABEL not in df.columns:
    raise ValueError(f"Label '{LABEL}' not found in CSV columns")
labels_np = df[LABEL].values
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=args.seed)
fold_aucs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels_np), 1):
    print(f"\n========== Fold {fold}/{N_FOLDS} ({LABEL}) ==========")

    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    # Sampler for balance
    class_counts = train_df[LABEL].value_counts().to_dict()  # {0:neg,1:pos}
    weights = train_df[LABEL].apply(lambda x: 1.0 / class_counts[x]).values
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)# WeightedRandomSampler for data imbalance

    train_ds = SingleLabelDataset(train_df, IMG_DIR, LABEL, train_tf)
    val_ds   = SingleLabelDataset(val_df,   IMG_DIR, LABEL, val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, )
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,    )

    model = ResNetBinary().to(DEVICE)
    print(model)
    pos_weight = torch.tensor([(class_counts[0] / class_counts[1])]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)# BCELoss for multilabel task

    optimizer = optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_MIN) # Cosine lr_scheduler, avoid underfitting
    scaler = torch.cuda.amp.GradScaler()

    best_auc, best_th = 0.0, 0.5
    for epoch in range(1, EPOCHS + 1):# 20 epoch per fold
        tr_loss, tr_acc, tr_prec, tr_rec, tr_auc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        va_loss, va_acc, va_prec, va_rec, va_auc, th = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        print(
            f"Ep{epoch:02d} LR {scheduler.get_last_lr()[0]:.2e} | "
            f"Train L {tr_loss:.3f} A {tr_acc:.3f} P {tr_prec:.3f} R {tr_rec:.3f} AUC {tr_auc:.3f} | "
            f"Val L {va_loss:.3f} A {va_acc:.3f} P {va_prec:.3f} R {va_rec:.3f} AUC {va_auc:.3f} | Th {th:.2f}"
        )

        if va_auc > best_auc: # save model with best AUC
            best_auc, best_th = va_auc, th
            NEW_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "threshold": best_th}, NEW_MODELS_DIR / f"best_{LABEL}_fold{fold}.pth")

    fold_aucs.append(best_auc)
    print(f"Best AUC fold {fold}: {best_auc:.3f}  (th={best_th:.2f})\n")

print("========== CV Summary ==========")
print("AUCs:", [f"{x:.3f}" for x in fold_aucs])
print(f"Mean AUC: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")
