"""
MoE (Mixture of Experts) multi-label training for all 8 retinal disease labels.

Architecture:
  - Shared frozen ConvNeXt_Base backbone
  - SparseMoE layer: top-K routing from 8 experts (one per label)
  - Per-label binary head

Usage:
  python training/train_moe.py
  python training/train_moe.py --epochs 30 --batch 16 --unfreeze_epoch 5
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.paths import NEW_MODELS_DIR, TRAIN_CSV, TRAIN_IMAGES_DIR

# ── args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv",            default=str(TRAIN_CSV))
parser.add_argument("--img_dir",        default=str(TRAIN_IMAGES_DIR / "merged"))
parser.add_argument("--epochs",         type=int,   default=20)
parser.add_argument("--batch",          type=int,   default=16)
parser.add_argument("--lr",             type=float, default=1e-4)
parser.add_argument("--lr_backbone",    type=float, default=1e-5)
parser.add_argument("--lr_min",         type=float, default=1e-6)
parser.add_argument("--unfreeze_epoch", type=int,   default=5,
                    help="epoch at which backbone is unfrozen (0 = always frozen)")
parser.add_argument("--n_experts",      type=int,   default=8)
parser.add_argument("--top_k",          type=int,   default=2)
parser.add_argument("--lb_coef",        type=float, default=0.01,
                    help="load-balance loss coefficient")
parser.add_argument("--seed",           type=int,   default=42)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

LABELS  = ["N", "D", "G", "C", "A", "H", "M", "O"]
N_LABELS = len(LABELS)
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_DIR = Path(args.img_dir)
MOE_DIR = NEW_MODELS_DIR / "moe"
MOE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}  |  experts={args.n_experts}  top_k={args.top_k}")


# ── dataset ──────────────────────────────────────────────────────────────────
class MultiLabelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row["ID"])
        img = Image.open(self.img_dir / f"{img_id}_merge.jpg").convert("RGB")
        labels = torch.tensor(row[LABELS].values.astype(np.float32))
        if self.transform:
            img = self.transform(img)
        return img, labels


train_tf = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── model ─────────────────────────────────────────────────────────────────────
class ExpertMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_in),
        )

    def forward(self, x):
        return self.net(x)


class SparseMoE(nn.Module):
    """Top-K sparse gating with load-balance auxiliary loss."""

    def __init__(self, d_model: int, n_experts: int, top_k: int):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([ExpertMLP(d_model) for _ in range(n_experts)])

    def forward(self, x):
        # x: (B, d_model)
        logits = self.gate(x)                           # (B, E)
        scores = F.softmax(logits, dim=-1)              # (B, E)

        topk_vals, topk_idx = scores.topk(self.top_k, dim=-1)  # (B, K)
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-6)

        out = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_ids = topk_idx[:, k]       # (B,)
            weights    = topk_vals[:, k]      # (B,)
            for e in range(self.n_experts):
                mask = (expert_ids == e)
                if mask.any():
                    out[mask] += weights[mask].unsqueeze(1) * self.experts[e](x[mask])

        # load-balance loss: encourage uniform expert usage
        avg_scores = scores.mean(dim=0)           # (E,)
        lb_loss = self.n_experts * (avg_scores * avg_scores).sum()

        return out, lb_loss


class ConvNeXtMoE(nn.Module):
    def __init__(self, n_experts: int = 8, top_k: int = 2):
        super().__init__()
        base = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        d_model = 1024
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.moe  = SparseMoE(d_model, n_experts, top_k)
        self.head = nn.Linear(d_model, N_LABELS)

    def forward(self, x):
        feat = self.backbone(x)[:, :, 0, 0]   # (B, 1024)
        feat = self.norm(feat)
        feat, lb_loss = self.moe(feat)
        logits = self.head(feat)               # (B, 8)
        return logits, lb_loss

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


# ── training helpers ──────────────────────────────────────────────────────────
def build_pos_weights(df: pd.DataFrame) -> torch.Tensor:
    weights = []
    for label in LABELS:
        pos = df[label].sum()
        neg = len(df) - pos
        weights.append(neg / max(pos, 1))
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def train_epoch(model, loader, optimizer, scaler, pos_weights, lb_coef):
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits, lb_loss = model(imgs)
            bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weights)
            loss = bce + lb_coef * lb_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        all_labels.append(labels.cpu().numpy())
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)
    aucs = []
    for i in range(N_LABELS):
        if all_labels[:, i].sum() > 0:
            aucs.append(roc_auc_score(all_labels[:, i], all_probs[:, i]))
    return total_loss / len(loader.dataset), float(np.mean(aucs)) if aucs else 0.0


def eval_epoch(model, loader, pos_weights):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits, lb_loss = model(imgs)
            bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weights)
            total_loss += bce.item() * imgs.size(0)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)
    aucs = []
    for i, label in enumerate(LABELS):
        if all_labels[:, i].sum() > 0:
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            aucs.append(auc)
            print(f"  {label}: AUC={auc:.3f}", end="  ")
    print()
    return total_loss / len(loader.dataset), float(np.mean(aucs)) if aucs else 0.0


# ── main ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(args.csv)
for label in LABELS:
    if label not in df.columns:
        raise ValueError(f"Label '{label}' not in CSV")

# 80/20 split (reproducible)
rng = np.random.default_rng(args.seed)
val_mask = rng.random(len(df)) < 0.2
train_df = df[~val_mask].reset_index(drop=True)
val_df   = df[val_mask].reset_index(drop=True)
print(f"Train: {len(train_df)}  Val: {len(val_df)}")

# sampler: weight by number of positive labels per sample (favour rarer multi-label)
label_counts = train_df[LABELS].sum().values          # (8,)
inv_freq     = 1.0 / (label_counts + 1)
sample_weights = train_df[LABELS].values.dot(inv_freq)
sample_weights = sample_weights / sample_weights.sum()
sampler = WeightedRandomSampler(sample_weights.tolist(), num_samples=len(train_df), replacement=True)

train_ds = MultiLabelDataset(train_df, IMG_DIR, train_tf)
val_ds   = MultiLabelDataset(val_df,   IMG_DIR, val_tf)
train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,    num_workers=4, pin_memory=True)

model = ConvNeXtMoE(n_experts=args.n_experts, top_k=args.top_k).to(DEVICE)
model.freeze_backbone()

pos_weights = build_pos_weights(train_df)

optimizer = optim.AdamW([
    {"params": model.backbone.parameters(), "lr": args.lr_backbone},
    {"params": list(model.norm.parameters()) + list(model.moe.parameters()) + list(model.head.parameters()), "lr": args.lr},
], weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
scaler = torch.cuda.amp.GradScaler()

best_auc = 0.0
for epoch in range(1, args.epochs + 1):
    if args.unfreeze_epoch > 0 and epoch == args.unfreeze_epoch:
        model.unfreeze_backbone()
        print(f">>> Epoch {epoch}: backbone unfrozen")

    tr_loss, tr_auc = train_epoch(model, train_loader, optimizer, scaler, pos_weights, args.lb_coef)
    va_loss, va_auc = eval_epoch(model, val_loader, pos_weights)
    scheduler.step()

    print(
        f"Ep{epoch:02d} LR {scheduler.get_last_lr()[0]:.2e} | "
        f"Train L={tr_loss:.4f} mAUC={tr_auc:.3f} | "
        f"Val   L={va_loss:.4f} mAUC={va_auc:.3f}"
    )

    if va_auc > best_auc:
        best_auc = va_auc
        torch.save(
            {"state_dict": model.state_dict(), "epoch": epoch, "val_auc": va_auc},
            MOE_DIR / "best_moe.pth",
        )
        print(f"  ✓ Saved best model  (mAUC={va_auc:.3f})")

print(f"\nDone. Best val mAUC: {best_auc:.3f}")
print(f"Checkpoint: {MOE_DIR / 'best_moe.pth'}")
