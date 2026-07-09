from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

try:
    import timm
except ImportError:  # pragma: no cover - surfaced when a model is instantiated.
    timm = None


LABELS = ["N", "D", "G", "C", "A", "H", "M", "O"]
IMG_SIZE = 512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class ExperimentConfig:
    preset: str
    model_name: str
    input_mode: str
    head: str
    img_size: int = IMG_SIZE
    hidden_dim: int = 512
    num_classes: int = len(LABELS)

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    "effnet_b4_linear": ExperimentConfig(
        preset="effnet_b4_linear",
        model_name="tf_efficientnet_b4.ns_jft_in1k",
        input_mode="merged",
        head="linear",
    ),
    "effnet_b4_label_corr": ExperimentConfig(
        preset="effnet_b4_label_corr",
        model_name="tf_efficientnet_b4.ns_jft_in1k",
        input_mode="merged",
        head="label_corr",
    ),
    "swin_tiny_linear": ExperimentConfig(
        preset="swin_tiny_linear",
        model_name="swin_tiny_patch4_window7_224.ms_in1k",
        input_mode="merged",
        head="linear",
    ),
    "swin_tiny_label_corr": ExperimentConfig(
        preset="swin_tiny_label_corr",
        model_name="swin_tiny_patch4_window7_224.ms_in1k",
        input_mode="merged",
        head="label_corr",
    ),
    "swin_tiny_ml_decoder": ExperimentConfig(
        preset="swin_tiny_ml_decoder",
        model_name="swin_tiny_patch4_window7_224.ms_in1k",
        input_mode="merged",
        head="ml_decoder",
    ),
    "swin_small_linear": ExperimentConfig(
        preset="swin_small_linear",
        model_name="swin_small_patch4_window7_224.ms_in1k",
        input_mode="merged",
        head="linear",
    ),
    "convnext_tiny_linear": ExperimentConfig(
        preset="convnext_tiny_linear",
        model_name="convnext_tiny",
        input_mode="merged",
        head="linear",
    ),
    "convnext_base_linear": ExperimentConfig(
        preset="convnext_base_linear",
        model_name="convnext_base",
        input_mode="merged",
        head="linear",
    ),
    "dual_branch_label_gcn_asl": ExperimentConfig(
        preset="dual_branch_label_gcn_asl",
        model_name="tf_efficientnet_b4.ns_jft_in1k",
        input_mode="binocular",
        head="label_gcn",
    ),
    "dual_branch_convnext_base_label_gcn_asl": ExperimentConfig(
        preset="dual_branch_convnext_base_label_gcn_asl",
        model_name="convnext_base",
        input_mode="binocular",
        head="label_gcn",
    ),
}


def create_timm_model(model_name: str, *, pretrained: bool, num_classes: int, img_size: int, global_pool: str | None = None):
    kwargs = {
        "pretrained": pretrained,
        "num_classes": num_classes,
    }
    if global_pool is not None:
        kwargs["global_pool"] = global_pool
    if model_name.startswith("swin_"):
        kwargs["img_size"] = img_size
    return timm.create_model(model_name, **kwargs)


def build_transforms(train: bool, img_size: int = IMG_SIZE):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.04),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class ODIRExperimentDataset(Dataset):
    def __init__(
        self,
        df_or_csv: pd.DataFrame | str | Path,
        image_root: str | Path,
        merged_dir: str | Path,
        input_mode: str,
        transform=None,
    ):
        self.df = pd.read_csv(df_or_csv) if not isinstance(df_or_csv, pd.DataFrame) else df_or_csv.copy()
        self.df = self.df.reset_index(drop=True)
        missing = [label for label in LABELS if label not in self.df.columns]
        if missing:
            raise ValueError(f"Missing label columns: {missing}")
        self.image_root = Path(image_root)
        self.merged_dir = Path(merged_dir)
        self.input_mode = input_mode
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _open(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        return self.transform(image) if self.transform else image

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        target = torch.from_numpy(row[LABELS].astype("float32").to_numpy())
        if self.input_mode == "merged":
            image = self._open(self.merged_dir / f"{row['ID']}_merge.jpg")
            return image, target
        if self.input_mode == "binocular":
            left = self._open(self.image_root / str(row["Left-Fundus"]))
            right = self._open(self.image_root / str(row["Right-Fundus"]))
            return left, right, target
        raise ValueError(f"Unsupported input_mode: {self.input_mode}")


class AsymmetricLoss(nn.Module):
    """ASL-style loss for long-tailed multi-label learning."""

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos = torch.sigmoid(logits)
        neg = 1.0 - pos
        if self.clip > 0:
            neg = (neg + self.clip).clamp(max=1.0)
        pos_loss = targets * torch.log(pos.clamp(min=1e-8))
        neg_loss = (1.0 - targets) * torch.log(neg.clamp(min=1e-8))
        pt = pos * targets + neg * (1.0 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        return -((pos_loss + neg_loss) * (1.0 - pt).pow(gamma)).mean()


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        return (loss * (1.0 - pt).pow(self.gamma)).mean()


class WeightedAsymmetricLoss(nn.Module):
    """ASL with per-label weights for class-balanced multi-label training."""

    def __init__(
        self,
        class_weight: torch.Tensor,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.register_buffer("class_weight", class_weight.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos = torch.sigmoid(logits)
        neg = 1.0 - pos
        if self.clip > 0:
            neg = (neg + self.clip).clamp(max=1.0)
        pos_loss = targets * torch.log(pos.clamp(min=1e-8))
        neg_loss = (1.0 - targets) * torch.log(neg.clamp(min=1e-8))
        pt = pos * targets + neg * (1.0 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        loss = -((pos_loss + neg_loss) * (1.0 - pt).pow(gamma))
        return (loss * self.class_weight.reshape(1, -1)).mean()


class DistributionBalancedLoss(nn.Module):
    """Lightweight DB-Loss inspired by long-tailed multi-label learning."""

    def __init__(
        self,
        pos_weight: torch.Tensor,
        neg_weight: torch.Tensor,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.register_buffer("pos_weight", pos_weight.float())
        self.register_buffer("neg_weight", neg_weight.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos = torch.sigmoid(logits)
        neg = 1.0 - pos
        if self.clip > 0:
            neg = (neg + self.clip).clamp(max=1.0)
        pos_loss = targets * torch.log(pos.clamp(min=1e-8))
        neg_loss = (1.0 - targets) * torch.log(neg.clamp(min=1e-8))
        pt = pos * targets + neg * (1.0 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        weights = targets * self.pos_weight.reshape(1, -1) + (1.0 - targets) * self.neg_weight.reshape(1, -1)
        return -((pos_loss + neg_loss) * (1.0 - pt).pow(gamma) * weights).mean()


class LinearMergedClassifier(nn.Module):
    def __init__(self, config: ExperimentConfig, pretrained: bool = True):
        super().__init__()
        if config.model_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.convnext_tiny(weights=weights)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, config.num_classes)
        elif config.model_name == "convnext_base":
            weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.convnext_base(weights=weights)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, config.num_classes)
        elif timm is None:
            raise ImportError("Install timm before training: pip install timm")
        else:
            self.backbone = create_timm_model(
                config.model_name,
                pretrained=pretrained,
                num_classes=config.num_classes,
                img_size=config.img_size,
            )
        self.config = config

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)


class LabelCorrelationHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, adjacency: torch.Tensor, dropout: float = 0.2):
        super().__init__()
        self.feature_norm = nn.LayerNorm(in_dim)
        self.feature_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.label_embed = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)
        self.graph_proj = nn.Linear(hidden_dim, hidden_dim)
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.register_buffer("adjacency", adjacency.float())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        image_state = self.feature_proj(self.feature_norm(features))
        label_state = torch.matmul(self.adjacency, self.label_embed)
        label_state = F.gelu(self.graph_proj(label_state))
        scale = float(image_state.shape[-1]) ** 0.5
        return image_state @ label_state.t() / scale + self.bias


class LabelCorrelationMergedClassifier(nn.Module):
    def __init__(self, config: ExperimentConfig, adjacency: torch.Tensor, pretrained: bool = True):
        super().__init__()
        if timm is None:
            raise ImportError("Install timm before training: pip install timm")
        self.backbone = create_timm_model(
            config.model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=config.img_size,
            global_pool="avg",
        )
        self.head = LabelCorrelationHead(self.backbone.num_features, config.hidden_dim, config.num_classes, adjacency)
        self.config = config

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(image))


class MLDecoderHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
        )
        self.label_queries = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)
        self.decoder = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.class_weight = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4 and features.shape[-1] >= features.shape[1]:
            tokens = features.reshape(features.shape[0], -1, features.shape[-1])
        elif features.ndim == 4:
            tokens = features.flatten(2).transpose(1, 2)
        else:
            tokens = features.unsqueeze(1)
        tokens = self.feature_proj(tokens)
        queries = self.label_queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        decoded, _ = self.decoder(queries, tokens, tokens, need_weights=False)
        decoded = self.norm(decoded)
        return (decoded * self.class_weight.unsqueeze(0)).sum(dim=-1) + self.bias


class MLDecoderMergedClassifier(nn.Module):
    def __init__(self, config: ExperimentConfig, pretrained: bool = True):
        super().__init__()
        if timm is None:
            raise ImportError("Install timm before training: pip install timm")
        self.backbone = create_timm_model(
            config.model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=config.img_size,
            global_pool="",
        )
        self.head = MLDecoderHead(self.backbone.num_features, config.hidden_dim, config.num_classes)
        self.config = config

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(image))


class LabelGraphHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, adjacency: torch.Tensor):
        super().__init__()
        self.image_proj = nn.Linear(in_dim, hidden_dim)
        self.label_embed = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)
        self.graph_proj = nn.Linear(hidden_dim, hidden_dim)
        self.register_buffer("adjacency", adjacency.float())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        image_state = self.image_proj(features)
        label_state = torch.matmul(self.adjacency, self.label_embed)
        label_state = torch.relu(self.graph_proj(label_state))
        return image_state @ label_state.t()


class DualBranchLabelGraphModel(nn.Module):
    def __init__(self, config: ExperimentConfig, adjacency: torch.Tensor, pretrained: bool = True):
        super().__init__()
        if config.model_name == "convnext_base":
            weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.convnext_base(weights=weights)
            self.backbone = nn.Sequential(*list(base.children())[:-1], nn.Flatten(1))
            feat_dim = base.classifier[-1].in_features
        elif config.model_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.convnext_tiny(weights=weights)
            self.backbone = nn.Sequential(*list(base.children())[:-1], nn.Flatten(1))
            feat_dim = base.classifier[-1].in_features
        elif timm is None:
            raise ImportError("Install timm before training: pip install timm")
        else:
            self.backbone = create_timm_model(
                config.model_name,
                pretrained=pretrained,
                num_classes=0,
                img_size=config.img_size,
                global_pool="avg",
            )
            feat_dim = self.backbone.num_features
        self.config = config
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 4, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid(),
        )
        self.head = LabelGraphHead(feat_dim, config.hidden_dim, config.num_classes, adjacency)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_feat = self.backbone(left)
        right_feat = self.backbone(right)
        fusion_input = torch.cat([left_feat, right_feat, torch.abs(left_feat - right_feat), left_feat * right_feat], dim=1)
        gate = self.gate(fusion_input)
        fused = gate * left_feat + (1.0 - gate) * right_feat
        return self.head(fused)


def build_label_graph(df: pd.DataFrame, smoothing: float = 1.0) -> torch.Tensor:
    labels = df[LABELS].astype("float32").to_numpy()
    cooccur = labels.T @ labels
    cooccur = cooccur + np.eye(len(LABELS), dtype=np.float32) * smoothing
    row_sum = np.clip(cooccur.sum(axis=1, keepdims=True), 1e-6, None)
    return torch.from_numpy(cooccur / row_sum).float()


def build_model(config: ExperimentConfig, label_graph: torch.Tensor | None = None, pretrained: bool = True) -> nn.Module:
    if config.head == "linear":
        return LinearMergedClassifier(config, pretrained=pretrained)
    if config.head == "label_corr":
        if label_graph is None:
            label_graph = torch.eye(config.num_classes)
        return LabelCorrelationMergedClassifier(config, label_graph, pretrained=pretrained)
    if config.head == "ml_decoder":
        return MLDecoderMergedClassifier(config, pretrained=pretrained)
    if config.head == "label_gcn":
        if label_graph is None:
            label_graph = torch.eye(config.num_classes)
        return DualBranchLabelGraphModel(config, label_graph, pretrained=pretrained)
    raise ValueError(f"Unsupported head: {config.head}")


def compute_pos_weight(df: pd.DataFrame) -> torch.Tensor:
    positives = df[LABELS].sum(axis=0).astype("float32").to_numpy()
    negatives = len(df) - positives
    return torch.from_numpy(negatives / np.clip(positives, 1.0, None)).float()


def compute_effective_class_weight(df: pd.DataFrame, beta: float = 0.9999) -> torch.Tensor:
    positives = df[LABELS].sum(axis=0).astype("float32").to_numpy()
    effective_num = 1.0 - np.power(beta, np.clip(positives, 1.0, None))
    weights = (1.0 - beta) / np.clip(effective_num, 1e-8, None)
    weights = weights / np.mean(weights)
    return torch.from_numpy(weights.astype("float32")).float()


def compute_distribution_balanced_weights(df: pd.DataFrame, beta: float = 0.9999) -> tuple[torch.Tensor, torch.Tensor]:
    labels = df[LABELS].astype("float32").to_numpy()
    positives = labels.sum(axis=0)
    class_freq = np.clip(positives / max(len(df), 1), 1e-6, 1.0)
    cardinality = np.clip(labels.sum(axis=1, keepdims=True), 1.0, None)
    rebalanced = ((1.0 / class_freq.reshape(1, -1)) * labels / cardinality).sum(axis=0)
    rebalanced = rebalanced / max(len(df), 1)
    effective = compute_effective_class_weight(df, beta=beta).numpy()
    pos_weight = np.sqrt(np.clip(rebalanced, 1e-3, None)) * effective
    pos_weight = pos_weight / np.mean(pos_weight)
    neg_weight = np.sqrt(np.clip(1.0 - class_freq, 1e-3, None))
    neg_weight = neg_weight / np.mean(neg_weight)
    return (
        torch.from_numpy(pos_weight.astype("float32")).float(),
        torch.from_numpy(neg_weight.astype("float32")).float(),
    )


def load_experiment_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ExperimentConfig(**ckpt["config"])
    label_graph = torch.tensor(ckpt.get("label_graph", np.eye(len(LABELS))), dtype=torch.float32)
    model = build_model(config, label_graph=label_graph, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    thresholds = np.array(ckpt.get("thresholds", [0.5] * len(LABELS)), dtype=np.float32)
    return model, thresholds, ckpt
