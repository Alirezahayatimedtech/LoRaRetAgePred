"""
simple_baseline.py
Xception baseline (via timm) for retinal age estimation.
Mimics the interface of RETFoundLoRAAgePred for easy swapping.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception as exc:  # pragma: no cover
    raise ImportError("timm is required for the Xception baseline") from exc


class AgePredictionHead(nn.Module):
    """Head similar to RETFound version, adapted for generic feature map size."""

    def __init__(self, in_channels: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        age_maps = self.conv2(x)
        age_predictions = F.adaptive_avg_pool2d(age_maps, 1).squeeze(-1).squeeze(-1)
        return age_predictions, age_maps


class SimpleXceptionAgePred(nn.Module):
    """Xception baseline with spatial head, interface-compatible with RETFoundLoRAAgePred."""

    def __init__(self, pretrained: bool = False, head_hidden_dim: int = 256, head_dropout: float = 0.2):
        super().__init__()
        # timm Xception returns feature maps when num_classes=0 and global_pool=""
        self.backbone = timm.create_model(
            "xception",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        self.backbone_channels = self.backbone.num_features
        self.head = AgePredictionHead(
            in_channels=self.backbone_channels,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )
        # Optional student-side projection head used only for feature distillation.
        self.distill_proj = None

    def extract_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def extract_image_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled per-image feature vectors [B, C] for distillation or simple heads."""
        feats = self.extract_spatial_features(x)
        if feats.ndim != 4:
            raise RuntimeError(f"Expected Xception spatial features [B,C,H,W], got {tuple(feats.shape)}")
        return F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)

    def forward(self, x: torch.Tensor):
        feats = self.extract_spatial_features(x)
        return self.head(feats)

    # Compatibility stubs (save/load full state since no LoRA)
    def save_lora_checkpoint(self, path: str):
        torch.save(self.state_dict(), path)

    def load_lora_checkpoint(self, path: str):
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state, strict=True)
