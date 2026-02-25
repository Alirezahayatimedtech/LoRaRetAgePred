import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import loralib as lora
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import PROJECT_ROOT

RETFOUND_MAE_DIR = PROJECT_ROOT / "RETFound_MAE"
if str(RETFOUND_MAE_DIR) not in sys.path:
    sys.path.insert(0, str(RETFOUND_MAE_DIR))

import models_vit as models  # noqa: E402
from util.pos_embed import interpolate_pos_embed  # noqa: E402


def _copy_linear_params(dst: nn.Module, src: nn.Module) -> None:
    """Copy pretrained linear params when wrapping a layer with LoRA."""
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if getattr(src, "bias", None) is not None and getattr(dst, "bias", None) is not None:
            dst.bias.copy_(src.bias)


def load_retfound_backbone_with_lora(
    ckpt_path: Path,
    img_size=256,
    global_pool=True,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_blocks: int = 4,
    lora_dropout: float = 0.0,
    enable_lora: bool = True,
    merge_lora_weights: bool = True,
) -> nn.Module:
    """
    Load RETFound backbone with LoRA adaptation for the last K transformer blocks
    """
    model = models.__dict__["RETFound_mae"](
        img_size=img_size, num_classes=1, drop_path_rate=0.0, global_pool=global_pool
    )

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        if "model" in state and isinstance(state["model"], dict):
            ckpt = state["model"]
        elif "teacher" in state and isinstance(state["teacher"], dict):
            ckpt = state["teacher"]
        elif "state_dict" in state and isinstance(state["state_dict"], dict):
            ckpt = state["state_dict"]
        else:
            ckpt = state
    else:
        ckpt = state

    new_sd = {}
    for k, v in ckpt.items():
        k = k.replace("module.", "").replace("backbone.", "")
        k = k.replace("mlp.w12.", "mlp.fc1.").replace("mlp.w3.", "mlp.fc2.")
        new_sd[k] = v

    st = model.state_dict()
    for k in ("head.weight", "head.bias"):
        if k in new_sd and k in st and new_sd[k].shape != st[k].shape:
            del new_sd[k]

    interpolate_pos_embed(model, new_sd)
    _ = model.load_state_dict(new_sd, strict=False)
    model.head = nn.Identity()

    if enable_lora:
        for param in model.parameters():
            param.requires_grad = False

        total_blocks = len(model.blocks)
        if lora_blocks > total_blocks:
            raise ValueError(f"Requested lora_blocks={lora_blocks} exceeds available blocks={total_blocks}")
        for i in range(total_blocks - lora_blocks, total_blocks):
            block = model.blocks[i]

            if hasattr(block.attn, 'qkv'):
                orig_qkv = block.attn.qkv
                lora_qkv = lora.MergedLinear(
                    orig_qkv.in_features,
                    orig_qkv.out_features,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=orig_qkv.bias is not None,
                    enable_lora=[True, False, True],  # q, k, v - apply to q and v only
                    fan_in_fan_out=False,
                    merge_weights=merge_lora_weights,
                )
                _copy_linear_params(lora_qkv, orig_qkv)
                block.attn.qkv = lora_qkv
            elif hasattr(block.attn, 'q_proj') and hasattr(block.attn, 'v_proj'):
                orig_q = block.attn.q_proj
                orig_v = block.attn.v_proj

                lora_q = lora.Linear(
                    orig_q.in_features,
                    orig_q.out_features,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=orig_q.bias is not None,
                    fan_in_fan_out=False,
                    merge_weights=merge_lora_weights,
                )
                lora_v = lora.Linear(
                    orig_v.in_features,
                    orig_v.out_features,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=orig_v.bias is not None,
                    fan_in_fan_out=False,
                    merge_weights=merge_lora_weights,
                )
                _copy_linear_params(lora_q, orig_q)
                _copy_linear_params(lora_v, orig_v)
                block.attn.q_proj = lora_q
                block.attn.v_proj = lora_v

        lora.mark_only_lora_as_trainable(model, bias='none')

    model.eval()
    return model


class AgePredictionHead(nn.Module):
    """Spatial head for age prediction that maintains spatial structure for saliency maps"""
    def __init__(self,
                 in_channels: int,
                 hidden_dim: int = 256,
                 dropout: float = 0.2,
                 upsample_factor: Optional[int] = 2):
        super().__init__()
        self.upsample_factor = upsample_factor

        if upsample_factor:
            self.upsample = nn.Upsample(scale_factor=upsample_factor, mode='bilinear', align_corners=False)
            conv_in_channels = in_channels
        else:
            self.upsample = None
            conv_in_channels = in_channels

        self.conv1 = nn.Conv2d(conv_in_channels, hidden_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.upsample:
            x = self.upsample(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        age_maps = self.conv2(x)
        age_predictions = F.adaptive_avg_pool2d(age_maps, 1).squeeze(-1).squeeze(-1)
        return age_predictions, age_maps


class InputPreAdapter(nn.Module):
    """
    Small residual adapter before RETFound patch embedding.

    Goal: absorb device / acquisition style shifts (brightness / contrast / texture statistics)
    while staying near-identity at initialization.
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 16):
        super().__init__()
        hidden_dim = int(max(4, hidden_dim))
        num_groups = 4 if hidden_dim % 4 == 0 else 1

        self.in_norm = nn.InstanceNorm2d(in_channels, affine=True, eps=1e-5)
        self.conv_in = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=True)
        self.dw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=True)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim)
        self.act = nn.GELU()
        self.conv_out = nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=True)
        # Learnable scalar gate keeps the adapter close to identity early in training.
        self.res_scale = nn.Parameter(torch.tensor(1.0))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.ones_(self.in_norm.weight)
        nn.init.zeros_(self.in_norm.bias)
        nn.init.kaiming_normal_(self.conv_in.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.conv_in.bias)
        nn.init.kaiming_normal_(self.dw.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.dw.bias)
        nn.init.ones_(self.gn.weight)
        nn.init.zeros_(self.gn.bias)
        # Exact identity at init (adapter delta = 0); conv_out learns first, then upstream layers follow.
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.in_norm(x)
        y = self.conv_in(y)
        y = self.act(y)
        y = self.dw(y)
        y = self.gn(y)
        y = self.act(y)
        y = self.conv_out(y)
        return residual + self.res_scale * y


class AttentionMILRegressor(nn.Module):
    """Attention-based MIL pooling + regression on per-image features."""

    def __init__(
        self,
        in_dim: int,
        attn_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        heteroscedastic: bool = False,
        logvar_min: float = -10.0,
        logvar_max: float = 6.0,
    ):
        super().__init__()
        attn_dim = int(max(16, attn_dim))
        hidden_dim = int(max(16, hidden_dim))
        self.heteroscedastic = bool(heteroscedastic)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.attn = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )
        self.reg = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.logvar_reg = None
        if self.heteroscedastic:
            self.logvar_reg = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, feats: torch.Tensor, return_pooled: bool = False, return_logvar: bool = False):
        if feats.ndim != 2:
            raise ValueError(f"Expected feats shape [N, D], got {tuple(feats.shape)}")
        if feats.shape[0] == 0:
            raise ValueError("MIL bag has zero instances")
        logits = self.attn(feats).squeeze(-1)  # [N]
        weights = torch.softmax(logits, dim=0)
        pooled = torch.sum(feats * weights.unsqueeze(-1), dim=0, keepdim=True)  # [1, D]
        pred = self.reg(pooled).view(-1)  # [1]
        logvar = None
        if return_logvar and self.logvar_reg is not None:
            logvar = self.logvar_reg(pooled).view(-1)
            logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)
        if return_pooled and return_logvar:
            return pred, weights, pooled, logvar
        if return_pooled:
            return pred, weights, pooled
        if return_logvar:
            return pred, weights, logvar
        return pred, weights


class RETFoundLoRAAgePred(nn.Module):
    """Complete RETFound + LoRA model for age prediction with built-in saliency maps"""

    def __init__(self,
                 ckpt_path: Path,
                 img_size: int = 256,
                 global_pool: bool = False,  # Must be False to keep spatial tokens
                 lora_rank: int = 8,
                 lora_alpha: float = 16.0,
                 lora_blocks: int = 4,
                 lora_dropout: float = 0.05,
                 head_hidden_dim: int = 256,
                 head_dropout: float = 0.2,
                 upsample_factor: Optional[int] = 2,
                 keep_spatial_tokens: bool = False,
                 use_pre_adapter: bool = False,
                 pre_adapter_hidden_dim: int = 16,
                 use_mil_attention: bool = False,
                 mil_attn_dim: int = 128,
                 mil_hidden_dim: int = 256,
                 heteroscedastic_regression: bool = False,
                 ordinal_num_bins: int = 0,
                 ordinal_aux_hidden_dim: int = 0,
                 use_regime_aux: bool = False,
                 regime_aux_hidden_dim: int = 0):
        super().__init__()
        self.keep_spatial_tokens = bool(keep_spatial_tokens)
        self.use_pre_adapter = bool(use_pre_adapter)
        self.use_mil_attention = bool(use_mil_attention)
        self.heteroscedastic_regression = bool(heteroscedastic_regression)
        self.configured_lora_blocks = int(max(0, lora_blocks))
        self.active_lora_blocks = int(max(0, lora_blocks))
        self.ordinal_num_bins = int(max(0, ordinal_num_bins))
        self.use_regime_aux = bool(use_regime_aux)

        self.backbone = load_retfound_backbone_with_lora(
            ckpt_path=ckpt_path,
            img_size=img_size,
            global_pool=global_pool,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_blocks=lora_blocks,
            lora_dropout=lora_dropout,
            enable_lora=True
        )

        self.pre_adapter = InputPreAdapter(in_channels=3, hidden_dim=pre_adapter_hidden_dim) if self.use_pre_adapter else None

        # Backbone channel dim is RETFound embed_dim; avoid dummy forward at init when possible.
        backbone_channels = getattr(self.backbone, "embed_dim", None)
        if backbone_channels is None and hasattr(self.backbone, "pos_embed") and self.backbone.pos_embed is not None:
            backbone_channels = int(self.backbone.pos_embed.shape[-1])
        if backbone_channels is None:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, img_size, img_size).to(next(self.backbone.parameters()).device)
                features = self.extract_spatial_features(dummy_input)
                backbone_channels = int(features.shape[1])

        self.head = AgePredictionHead(
            in_channels=backbone_channels,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
            upsample_factor=upsample_factor
        )
        self.mil_head = None
        self.ordinal_head = None
        self.regime_head = None
        if self.use_mil_attention:
            self.mil_head = AttentionMILRegressor(
                in_dim=backbone_channels,
                attn_dim=mil_attn_dim,
                hidden_dim=mil_hidden_dim,
                dropout=head_dropout,
                heteroscedastic=self.heteroscedastic_regression,
            )
            # MIL mode uses attention pooling head instead of the spatial age head.
            for p in self.head.parameters():
                p.requires_grad = False

        if self.ordinal_num_bins >= 2:
            aux_hidden = int(max(16, ordinal_aux_hidden_dim or head_hidden_dim))
            self.ordinal_head = nn.Sequential(
                nn.Linear(backbone_channels, aux_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(head_dropout),
                nn.Linear(aux_hidden, self.ordinal_num_bins - 1),
            )
        if self.use_mil_attention and self.use_regime_aux:
            reg_hidden = int(max(16, regime_aux_hidden_dim or head_hidden_dim))
            self.regime_head = nn.Sequential(
                nn.Linear(backbone_channels, reg_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(head_dropout),
                nn.Linear(reg_hidden, 1),
            )

    @staticmethod
    def _set_block_lora_trainable(block: nn.Module, enabled: bool) -> bool:
        """Enable/disable LoRA delta params inside one transformer block."""
        changed_any = False
        for mod in block.modules():
            if isinstance(mod, (lora.Linear, lora.MergedLinear)):
                for name, p in mod.named_parameters(recurse=False):
                    if name.startswith("lora_"):
                        p.requires_grad = bool(enabled)
                        changed_any = True
        return changed_any

    def set_active_lora_blocks(self, active_blocks: int) -> int:
        """
        Freeze/unfreeze LoRA params by transformer block.

        The model must be built with LoRA injected into the last `configured_lora_blocks`
        blocks. This method activates only the last `active_blocks` of those injected
        blocks and keeps earlier injected blocks frozen.
        """
        if not hasattr(self, "backbone") or not hasattr(self.backbone, "blocks"):
            return 0
        total_blocks = len(self.backbone.blocks)
        max_active = int(min(max(0, active_blocks), self.configured_lora_blocks, total_blocks))
        start_injected = total_blocks - self.configured_lora_blocks
        start_active = total_blocks - max_active

        n_toggled = 0
        for idx, block in enumerate(self.backbone.blocks):
            if idx < start_injected:
                continue  # no LoRA modules injected here
            enable = idx >= start_active
            if self._set_block_lora_trainable(block, enable):
                n_toggled += 1
        self.active_lora_blocks = max_active
        return max_active

    def extract_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_adapter is not None:
            x = self.pre_adapter(x)
        x = self.backbone.patch_embed(x)

        cls_token = getattr(self.backbone, 'cls_token', None)
        if cls_token is not None:
            cls_tok = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tok, x), dim=1)

        if hasattr(self.backbone, 'pos_embed'):
            if self.backbone.pos_embed is not None:
                x = x + self.backbone.pos_embed
        if hasattr(self.backbone, 'pos_drop'):
            x = self.backbone.pos_drop(x)

        for blk in self.backbone.blocks:
            x = blk(x)

        if hasattr(self.backbone, 'norm'):
            x = self.backbone.norm(x)

        if self.keep_spatial_tokens:
            x = x[:, 1:]
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            if H * W != N:
                raise RuntimeError(f"Non-square token grid: N={N}, cannot reshape to HxW patch map")
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            return x

        # CLS-only features for scalar regression (better inductive bias for age prediction).
        cls = x[:, 0, :]
        return cls.unsqueeze(-1).unsqueeze(-1)

    def extract_image_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-image feature vectors [B, D] for MIL or simple pooled heads."""
        feats = self.extract_spatial_features(x)
        if feats.ndim == 4:
            if feats.shape[-2:] == (1, 1):
                return feats.squeeze(-1).squeeze(-1)
            return F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
        if feats.ndim == 2:
            return feats
        raise RuntimeError(f"Unexpected feature shape from backbone: {tuple(feats.shape)}")

    def mil_predict_from_features(
        self,
        feats: torch.Tensor,
        return_pooled: bool = False,
        return_logvar: bool = False,
    ):
        if self.mil_head is None:
            raise RuntimeError("MIL head is not enabled. Build model with use_mil_attention=True.")
        return self.mil_head(feats, return_pooled=return_pooled, return_logvar=return_logvar)

    def ordinal_logits_from_pooled_features(self, pooled_feats: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Predict ordinal logits from pooled per-bag feature vectors [B, D].

        Returns None when ordinal auxiliary head is disabled.
        """
        if self.ordinal_head is None:
            return None
        if pooled_feats.ndim == 3 and pooled_feats.shape[1] == 1:
            pooled_feats = pooled_feats.squeeze(1)
        if pooled_feats.ndim != 2:
            raise ValueError(f"Expected pooled_feats shape [B, D], got {tuple(pooled_feats.shape)}")
        return self.ordinal_head(pooled_feats)

    def regime_logits_from_pooled_features(self, pooled_feats: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Predict binary regime logits (young vs old age regime) from pooled bag features [B, D].

        Returns None when the regime auxiliary head is disabled.
        """
        if self.regime_head is None:
            return None
        if pooled_feats.ndim == 3 and pooled_feats.shape[1] == 1:
            pooled_feats = pooled_feats.squeeze(1)
        if pooled_feats.ndim != 2:
            raise ValueError(f"Expected pooled_feats shape [B, D], got {tuple(pooled_feats.shape)}")
        return self.regime_head(pooled_feats).view(-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_spatial_features(x)
        age_predictions, age_maps = self.head(features)
        return age_predictions, age_maps

    def get_age_saliency_maps(self, x: torch.Tensor) -> torch.Tensor:
        if not self.keep_spatial_tokens:
            raise RuntimeError(
                "Spatial saliency maps require keep_spatial_tokens=True. "
                "CLS-only mode does not produce meaningful spatial age maps."
            )
        self.eval()
        with torch.no_grad():
            age_predictions, age_maps = self.forward(x)
        saliency_maps = age_maps
        # Normalize per-map over spatial dimensions (H, W)
        B, C, H, W = saliency_maps.shape
        flat = saliency_maps.view(B, C, -1)
        min_vals = flat.min(dim=2, keepdim=True).values.view(B, C, 1, 1)
        max_vals = flat.max(dim=2, keepdim=True).values.view(B, C, 1, 1)
        saliency_maps = saliency_maps - min_vals
        saliency_maps = saliency_maps / (max_vals - min_vals + 1e-8)
        saliency_maps = F.interpolate(
            saliency_maps,
            size=(x.shape[2], x.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        return saliency_maps

    def save_lora_checkpoint(self, path: str):
        """Save LoRA parameters plus regression head weights."""
        state = {
            "backbone_lora": lora.lora_state_dict(self.backbone, bias='none'),
            "head": self.head.state_dict(),
        }
        if self.pre_adapter is not None:
            state["pre_adapter"] = self.pre_adapter.state_dict()
        if self.mil_head is not None:
            state["mil_head"] = self.mil_head.state_dict()
        if self.ordinal_head is not None:
            state["ordinal_head"] = self.ordinal_head.state_dict()
        if self.regime_head is not None:
            state["regime_head"] = self.regime_head.state_dict()
        torch.save(state, path)

    def load_lora_checkpoint(self, path: str):
        """
        Load LoRA checkpoint into the model (backbone + head).
        Supports old-format files that only contained backbone LoRA weights.
        """
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "backbone_lora" in checkpoint:
            # Ensure LoRA modules are in unmerged mode before loading deltas.
            self.backbone.train()
            self.backbone.load_state_dict(checkpoint["backbone_lora"], strict=False)
            # Re-merge loaded deltas for eval inference.
            self.backbone.eval()
            if self.pre_adapter is not None:
                if "pre_adapter" in checkpoint:
                    self.pre_adapter.load_state_dict(checkpoint["pre_adapter"], strict=False)
                else:
                    print("[WARN] Checkpoint missing pre_adapter weights; pre-adapter remains randomly initialized.")
            if self.mil_head is not None:
                if "mil_head" in checkpoint:
                    self.mil_head.load_state_dict(checkpoint["mil_head"], strict=False)
                else:
                    print("[WARN] Checkpoint missing mil_head weights; MIL head remains randomly initialized.")
            if self.ordinal_head is not None:
                if "ordinal_head" in checkpoint:
                    self.ordinal_head.load_state_dict(checkpoint["ordinal_head"], strict=False)
                else:
                    print("[WARN] Checkpoint missing ordinal_head weights; ordinal aux head remains randomly initialized.")
            if self.regime_head is not None:
                if "regime_head" in checkpoint:
                    self.regime_head.load_state_dict(checkpoint["regime_head"], strict=False)
                else:
                    print("[WARN] Checkpoint missing regime_head weights; regime aux head remains randomly initialized.")
            if "head" in checkpoint:
                self.head.load_state_dict(checkpoint["head"])
            else:
                print("[WARN] Checkpoint missing head weights; head remains randomly initialized.")
        else:
            print("[WARN] Loading legacy LoRA-only checkpoint; head remains randomly initialized.")
            self.backbone.train()
            self.backbone.load_state_dict(checkpoint, strict=False)
            self.backbone.eval()
