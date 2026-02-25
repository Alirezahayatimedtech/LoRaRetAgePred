"""Training and evaluation engine for RETFound LoRA age regression."""

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import DAY_WHITELIST, IMAGE_TYPES, COHORTS_TO_KEEP
from preprocess_age_lora import prepare_data  # for types only
from data_prep_age_lora import load_metadata
from bias_correction import apply_correction, apply_poly_correction


def mixup_data(x, y, alpha: float, device: str):
    """Standard mixup on images/targets."""
    if alpha <= 0:
        return x, y, y, 1.0, None
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index


def cutmix_data(x, y, alpha: float, device: str):
    """CutMix augmentation."""
    if alpha <= 0:
        return x, y, y, 1.0, None, None
    lam = np.random.beta(alpha, alpha)
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size, device=device)

    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)

    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (w * h + 1e-6))

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index


class Trainer:
    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        self.loss_fn = nn.SmoothL1Loss(beta=1.0, reduction="none")
        self._ordinal_bins_cache = {}
        self._warned_ordinal_target_mismatch = False
        # Optional feature distillation (e.g., Xception student <- frozen RETFound teacher).
        self.distill_teacher = None
        self.distill_alpha = 0.0
        self._warned_distill_unsupported = False

    @staticmethod
    def _heteroscedastic_enabled(args) -> bool:
        return bool(getattr(args, "heteroscedastic_regression", False))

    @staticmethod
    def _hetero_norm_params(args) -> Tuple[float, float]:
        mean = float(getattr(args, "hetero_target_mean", 0.0) or 0.0)
        std = float(getattr(args, "hetero_target_std", 1.0) or 1.0)
        if (not np.isfinite(std)) or std <= 1e-6:
            std = 1.0
        return mean, std

    @classmethod
    def _hetero_age_to_z(cls, x: torch.Tensor, args) -> torch.Tensor:
        mean, std = cls._hetero_norm_params(args)
        return (x - mean) / std

    @classmethod
    def _hetero_z_to_age(cls, x: torch.Tensor, args) -> torch.Tensor:
        mean, std = cls._hetero_norm_params(args)
        return x * std + mean

    @classmethod
    def _hetero_logvar_z_to_age(cls, log_var_z: torch.Tensor, args) -> torch.Tensor:
        _, std = cls._hetero_norm_params(args)
        return log_var_z + (2.0 * math.log(std))

    @staticmethod
    def _gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor, reduction: str = "none") -> torch.Tensor:
        """
        Heteroscedastic Gaussian NLL for scalar regression:
          0.5 * (log_var + (y-mu)^2 / exp(log_var))
        """
        mu = mu.view(-1)
        target = target.view(-1)
        log_var = log_var.view(-1)
        inv_var = torch.exp(-log_var)
        loss = 0.5 * (log_var + (target - mu) ** 2 * inv_var)
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        return loss

    @staticmethod
    def _group_keys(batch, days, aggregate_by_rat: bool = False):
        eyes_list = [str(e) for e in batch["eye"]]
        rats = batch["rat_id"]
        if aggregate_by_rat:
            return [(r, float(d.item())) for r, d in zip(rats, days)]
        return [(r, e, float(d.item())) for r, e, d in zip(rats, eyes_list, days)]

    @staticmethod
    def _apply_skew(raw_loss: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor, args):
        # Skew disabled: use plain Smooth L1 (Huber)
        return raw_loss

    def _feature_distill_enabled(self, args) -> bool:
        if self.distill_teacher is None:
            return False
        if float(getattr(self, "distill_alpha", 0.0) or 0.0) <= 0:
            return False
        if args is None:
            return False
        if bool(getattr(args, "mil_attention", False)):
            return False
        if str(getattr(args, "model_type", "")).lower() != "xception":
            return False
        return hasattr(self.model, "distill_proj") and getattr(self.model, "distill_proj", None) is not None

    def _feature_distill_loss(self, imgs: torch.Tensor, student_feats: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        Feature-level distillation for the Xception baseline:
        - Student: pooled Xception features -> projection head
        - Teacher: frozen RETFound pooled features
        - Loss: MSE on L2-normalized features
        """
        if self.distill_teacher is None or getattr(self.model, "distill_proj", None) is None:
            return None
        # Student features
        if not hasattr(self.model, "extract_image_features"):
            return None
        if student_feats is None:
            student_feats = self.model.extract_image_features(imgs)  # [B, D_s]
        proj = self.model.distill_proj(student_feats)            # [B, D_t]
        # Frozen teacher features
        with torch.no_grad():
            teacher_feats = self.distill_teacher.extract_image_features(imgs)
        if teacher_feats.ndim != 2 or proj.ndim != 2:
            return None
        if teacher_feats.shape != proj.shape:
            return None
        s_norm = nn.functional.normalize(proj, dim=1)
        t_norm = nn.functional.normalize(teacher_feats, dim=1)
        return nn.functional.mse_loss(s_norm, t_norm, reduction="mean")

    def _ordinal_aux_enabled(self, args) -> bool:
        if args is None:
            return False
        if not bool(getattr(args, "ordinal_aux", False)):
            return False
        if float(getattr(args, "ordinal_aux_weight", 0.0) or 0.0) <= 0:
            return False
        return hasattr(self.model, "ordinal_head") and getattr(self.model, "ordinal_head", None) is not None

    def _regime_aux_enabled(self, args) -> bool:
        if args is None:
            return False
        if not bool(getattr(args, "regime_aux", False)):
            return False
        if float(getattr(args, "regime_aux_weight", 0.0) or 0.0) <= 0:
            return False
        return hasattr(self.model, "regime_head") and getattr(self.model, "regime_head", None) is not None

    def _get_ordinal_bins_tensor(self, args) -> Optional[torch.Tensor]:
        vals = getattr(args, "ordinal_bin_values_resolved", None)
        if not vals:
            return None
        key = tuple(float(v) for v in vals)
        t = self._ordinal_bins_cache.get(key)
        if t is None or t.device != self.device:
            t = torch.tensor(key, dtype=torch.float32, device=self.device)
            self._ordinal_bins_cache[key] = t
        return t

    def _ordinal_aux_loss(self, ordinal_logits: torch.Tensor, targets: torch.Tensor, args) -> Optional[torch.Tensor]:
        """
        CORAL-style ordinal auxiliary loss over discrete age bins.

        `ordinal_logits`: [B, K-1], `targets`: [B] in age units.
        """
        if ordinal_logits is None or ordinal_logits.numel() == 0:
            return None
        bins = self._get_ordinal_bins_tensor(args)
        if bins is None or bins.numel() < 2:
            return None
        if ordinal_logits.ndim != 2:
            return None
        num_bins = int(bins.numel())
        if int(ordinal_logits.shape[-1]) != num_bins - 1:
            return None

        tgt = targets.view(-1).to(self.device, dtype=torch.float32)
        # Map to the nearest known age bin (robust to float formatting noise).
        d = torch.abs(tgt.unsqueeze(1) - bins.unsqueeze(0))
        class_idx = torch.argmin(d, dim=1)
        min_diff = d.gather(1, class_idx.unsqueeze(1)).squeeze(1)
        if (not self._warned_ordinal_target_mismatch) and torch.any(min_diff > 0.5):
            self._warned_ordinal_target_mismatch = True
            bad = float(torch.max(min_diff).detach().cpu().item())
            print(f"[ORD] Warning: target ages do not exactly match ordinal bins (max nearest-bin diff={bad:.3f}). Using nearest-bin mapping.")

        thresholds = torch.arange(num_bins - 1, device=self.device).unsqueeze(0)  # [1, K-1]
        ordinal_targets = (class_idx.unsqueeze(1) > thresholds).to(ordinal_logits.dtype)
        loss = nn.functional.binary_cross_entropy_with_logits(ordinal_logits, ordinal_targets, reduction="mean")
        return loss

    def _regime_aux_loss(self, regime_logits: torch.Tensor, targets: torch.Tensor, args) -> Optional[torch.Tensor]:
        """
        Binary auxiliary loss for coarse age regime classification from pooled MIL features.

        Default regime split: young <= threshold (0), old > threshold (1).
        """
        if regime_logits is None or regime_logits.numel() == 0:
            return None
        if regime_logits.ndim != 1:
            regime_logits = regime_logits.view(-1)
        tgt = targets.view(-1).to(self.device, dtype=torch.float32)
        if tgt.numel() != regime_logits.numel():
            return None
        th = float(getattr(args, "regime_aux_age_threshold", 180.0) or 180.0)
        regime_targets = (tgt > th).to(regime_logits.dtype)
        return nn.functional.binary_cross_entropy_with_logits(regime_logits, regime_targets, reduction="mean")

    def _mil_predict_batch(self, batch, return_pooled: bool = False, return_logvar: bool = False):
        """MIL forward for a collated bag batch (bag = rat_id/eye/day)."""
        bags = batch.get("bags", [])
        if not bags:
            empty_preds = torch.empty(0, device=self.device)
            empty_pooled = torch.empty(0, 0, device=self.device) if return_pooled else None
            empty_logv = torch.empty(0, device=self.device) if return_logvar else None
            if return_logvar:
                return empty_preds, [], empty_pooled, empty_logv
            return empty_preds, [], empty_pooled
        bag_sizes = [int(b.shape[0]) for b in bags]
        all_imgs = torch.cat([b.to(self.device, non_blocking=True) for b in bags], dim=0)
        feats = self.model.extract_image_features(all_imgs)
        feat_splits = torch.split(feats, bag_sizes, dim=0)
        preds = []
        attn_weights = []
        pooled_feats = []
        logvars = []
        for f in feat_splits:
            if return_pooled and return_logvar:
                p, w, pooled, lv = self.model.mil_predict_from_features(f, return_pooled=True, return_logvar=True)
                pooled_feats.append(pooled.view(1, -1))
                logvars.append(lv.view(-1)[0] if lv is not None else torch.zeros_like(p.view(-1)[0]))
            elif return_pooled:
                p, w, pooled = self.model.mil_predict_from_features(f, return_pooled=True)
                pooled_feats.append(pooled.view(1, -1))
            elif return_logvar:
                p, w, lv = self.model.mil_predict_from_features(f, return_logvar=True)
                logvars.append(lv.view(-1)[0] if lv is not None else torch.zeros_like(p.view(-1)[0]))
            else:
                p, w = self.model.mil_predict_from_features(f)
            preds.append(p.view(-1)[0])
            attn_weights.append(w)
        pooled_cat = torch.cat(pooled_feats, dim=0) if return_pooled and pooled_feats else None
        preds_cat = torch.stack(preds, dim=0)
        if return_logvar:
            logvars_cat = torch.stack(logvars, dim=0) if logvars else torch.zeros_like(preds_cat)
            return preds_cat, attn_weights, pooled_cat, logvars_cat
        return preds_cat, attn_weights, pooled_cat

    def _mil_control_inter_eye_consistency(self, preds: torch.Tensor, batch, args):
        """
        Control-only OD/OS consistency penalty for MIL bags.

        Pairs bags within the current batch using (rat_id, day) and penalizes prediction
        differences between OD and OS when both eyes are present. This is batch-local and
        therefore stronger when the MIL bag batch size is larger.
        """
        lam = float(getattr(args, "mil_control_inter_eye_lambda", 0.0) or 0.0)
        if lam <= 0 or preds.numel() == 0:
            return None, 0
        if not isinstance(batch, dict):
            return None, 0
        if any(k not in batch for k in ("group", "rat_id", "eye", "day")):
            return None, 0

        def _norm_group(g):
            return str(g).strip().lower()

        def _norm_eye(e):
            return str(e).strip().upper()

        groups = batch.get("group", [])
        rats = batch.get("rat_id", [])
        eyes = batch.get("eye", [])
        days = batch.get("day")
        if days is None:
            return None, 0
        if torch.is_tensor(days):
            day_vals = [float(d.item()) for d in days]
        else:
            day_vals = [float(d) for d in days]

        paired_diffs = []
        buckets = {}
        n = min(len(groups), len(rats), len(eyes), len(day_vals), int(preds.numel()))
        for i in range(n):
            if _norm_group(groups[i]) != "controls":
                continue
            eye = _norm_eye(eyes[i])
            if eye not in ("OD", "OS"):
                continue
            # Round day to avoid float-key noise while preserving intended day bins.
            day_key = int(round(day_vals[i])) if math.isfinite(day_vals[i]) else day_vals[i]
            key = (str(rats[i]), day_key)
            buckets.setdefault(key, {}).setdefault(eye, []).append(i)

        for eye_map in buckets.values():
            if "OD" not in eye_map or "OS" not in eye_map:
                continue
            od_pred = preds[torch.as_tensor(eye_map["OD"], device=preds.device)].mean()
            os_pred = preds[torch.as_tensor(eye_map["OS"], device=preds.device)].mean()
            paired_diffs.append(od_pred - os_pred)

        if not paired_diffs:
            return None, 0

        diffs = torch.stack(paired_diffs, dim=0)
        mode = str(getattr(args, "mil_control_inter_eye_loss", "l1")).lower()
        if mode == "smoothl1":
            penalty = nn.functional.smooth_l1_loss(
                diffs,
                torch.zeros_like(diffs),
                reduction="mean",
                beta=1.0,
            )
        else:
            penalty = torch.mean(torch.abs(diffs))
        return penalty, int(diffs.numel())

    def _mil_control_day_loss_weights(self, batch, args, ref: torch.Tensor) -> torch.Tensor:
        """
        Optional MIL train-time per-bag weights for control day-specific emphasis.

        Current use: upweight control day 90 bags to improve control day-90 accuracy
        and reduce OD/OS inconsistency tails without changing the evaluation protocol.
        """
        w90 = float(getattr(args, "mil_control_day90_weight", 1.0) or 1.0)
        if w90 <= 1.0:
            return torch.ones_like(ref)
        if not isinstance(batch, dict):
            return torch.ones_like(ref)
        groups = batch.get("group", [])
        days = batch.get("day", None)
        if days is None:
            return torch.ones_like(ref)
        if torch.is_tensor(days):
            day_vals = days.to(ref.device, dtype=torch.float32).view(-1)
        else:
            try:
                day_vals = torch.as_tensor([float(d) for d in days], dtype=torch.float32, device=ref.device).view(-1)
            except Exception:
                return torch.ones_like(ref)
        n = min(int(ref.numel()), len(groups), int(day_vals.numel()))
        if n <= 0:
            return torch.ones_like(ref)
        weights = torch.ones_like(ref)
        for i in range(n):
            g = str(groups[i]).strip().lower()
            if g != "controls":
                continue
            # Day values are nominal integers but stored as float.
            if abs(float(day_vals[i].item()) - 90.0) <= 0.5:
                weights[i] = w90
        return weights

    def train_one_epoch(self, loader, optimizer, args) -> float:
        self.model.train()
        total_loss = 0.0
        steps = 0
        for batch in loader:
            if batch is None:
                continue
            if getattr(args, "mil_attention", False):
                use_ord = self._ordinal_aux_enabled(args)
                use_regime = self._regime_aux_enabled(args)
                use_hetero = self._heteroscedastic_enabled(args)
                if use_hetero:
                    preds, _, pooled_feats, pred_log_var = self._mil_predict_batch(
                        batch,
                        return_pooled=(use_ord or use_regime),
                        return_logvar=True,
                    )
                else:
                    preds, _, pooled_feats = self._mil_predict_batch(batch, return_pooled=(use_ord or use_regime))
                    pred_log_var = None
                targets_clean = batch["age_days"].to(self.device, non_blocking=True).view(-1)
                targets = targets_clean
                if args.label_noise_std > 0:
                    targets = targets + torch.randn_like(targets) * args.label_noise_std
                if use_hetero and pred_log_var is not None:
                    targets_z = self._hetero_age_to_z(targets, args)
                    raw_nll = self._gaussian_nll(preds, pred_log_var, targets_z, reduction="none")
                    weights = self._mil_control_day_loss_weights(batch, args, raw_nll)
                    denom = torch.clamp(weights.sum(), min=1e-6)
                    loss = torch.sum(raw_nll * weights) / denom
                    logvar_reg_w = float(getattr(args, "hetero_logvar_reg_weight", 0.0) or 0.0)
                    if logvar_reg_w > 0:
                        loss = loss + logvar_reg_w * torch.mean(pred_log_var ** 2)
                else:
                    raw_loss = self.loss_fn(preds, targets)
                    raw_loss = self._apply_skew(raw_loss, preds, targets, args)
                    weights = self._mil_control_day_loss_weights(batch, args, raw_loss)
                    denom = torch.clamp(weights.sum(), min=1e-6)
                    loss = torch.sum(raw_loss * weights) / denom
                if use_ord and pooled_feats is not None:
                    ord_logits = self.model.ordinal_logits_from_pooled_features(pooled_feats)
                    ord_loss = self._ordinal_aux_loss(ord_logits, targets_clean, args)
                    if ord_loss is not None:
                        loss = loss + float(getattr(args, "ordinal_aux_weight", 0.0)) * ord_loss
                if use_regime and pooled_feats is not None:
                    regime_logits = self.model.regime_logits_from_pooled_features(pooled_feats)
                    regime_loss = self._regime_aux_loss(regime_logits, targets_clean, args)
                    if regime_loss is not None:
                        loss = loss + float(getattr(args, "regime_aux_weight", 0.0)) * regime_loss
                preds_for_cons = self._hetero_z_to_age(preds, args) if use_hetero else preds
                cons_penalty, _ = self._mil_control_inter_eye_consistency(preds_for_cons, batch, args)
                if cons_penalty is not None:
                    loss = loss + float(getattr(args, "mil_control_inter_eye_lambda", 0.0)) * cons_penalty

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                steps += 1
                continue
            imgs = batch["image"].to(self.device, non_blocking=True)
            targets = batch["age_days"].to(self.device, non_blocking=True)
            days = batch["day"].to(self.device, non_blocking=True)

            # Early fusion: average images per rat(/eye)/day before backbone
            if args.early_fusion:
                keys = self._group_keys(batch, days, aggregate_by_rat=getattr(args, "aggregate_by_rat", False))
                grouped = {}
                for i, k in enumerate(keys):
                    grouped.setdefault(k, []).append(i)
                fused_imgs = []
                fused_targets = []
                fused_days = []
                for idxs in grouped.values():
                    fused_imgs.append(imgs[idxs].mean(dim=0, keepdim=False))
                    fused_targets.append(targets[idxs].mean())
                    fused_days.append(days[idxs].mean())
                imgs = torch.stack(fused_imgs, dim=0)
                targets = torch.stack(fused_targets, dim=0)
                days = torch.stack(fused_days, dim=0)

            use_cutmix = (args.cutmix_alpha > 0) and (np.random.rand() < args.cutmix_prob) and (not args.aggregate_features) and (not args.early_fusion)
            use_mix = False  # prefer CutMix; disable mixup
            if use_cutmix:
                imgs_m, ta, tb, lam, idx = cutmix_data(imgs, targets, alpha=args.cutmix_alpha, device=self.device)
                preds, _ = self.model(imgs_m)
                preds = preds.view(-1)
                ta = ta.view(-1); tb = tb.view(-1)
                da = days.view(-1); db = da[idx] if idx is not None else da
                wa = torch.ones_like(da)
                wb = torch.ones_like(db)
                raw_a = self.loss_fn(preds, ta) * wa
                raw_a = self._apply_skew(raw_a, preds, ta, args)
                raw_b = self.loss_fn(preds, tb) * wb
                raw_b = self._apply_skew(raw_b, preds, tb, args)
                loss = torch.mean(lam * raw_a + (1 - lam) * raw_b)
            else:
                if args.aggregate_features and not args.early_fusion:
                    feats = self.model.extract_spatial_features(imgs)
                    keys = self._group_keys(batch, days, aggregate_by_rat=getattr(args, "aggregate_by_rat", False))
                    grouped = {}
                    for i, k in enumerate(keys):
                        grouped.setdefault(k, []).append(i)
                    feat_means = []
                    tgt_means = []
                    day_means = []
                    for idxs in grouped.values():
                        feat_means.append(feats[idxs].mean(dim=0, keepdim=False))
                        tgt_means.append(targets[idxs].mean())
                        day_means.append(days[idxs].mean())
                    feats_cat = torch.stack(feat_means, dim=0)
                    targets = torch.stack(tgt_means, dim=0)
                    days_group = torch.stack(day_means, dim=0)
                    preds, _ = self.model.head(feats_cat)
                    preds = preds.view(-1)
                    weights = torch.ones_like(targets)
                    raw_loss = self.loss_fn(preds, targets)
                    raw_loss = self._apply_skew(raw_loss, preds, targets, args)
                    loss = torch.mean(raw_loss * weights)
                else:
                    distill_loss = None
                    if self._feature_distill_enabled(args):
                        # Avoid duplicate backbone passes when distilling Xception features.
                        feats_spatial = self.model.extract_spatial_features(imgs)
                        preds, _ = self.model.head(feats_spatial)
                        preds = preds.view(-1)
                        student_feats = nn.functional.adaptive_avg_pool2d(feats_spatial, 1).squeeze(-1).squeeze(-1)
                        distill_loss = self._feature_distill_loss(imgs, student_feats=student_feats)
                    else:
                        preds, _ = self.model(imgs)
                        preds = preds.view(-1)
                    targets = targets.view(-1)
                    if args.late_fusion:
                        keys = self._group_keys(batch, days, aggregate_by_rat=getattr(args, "aggregate_by_rat", False))
                        grouped = {}
                        for i, k in enumerate(keys):
                            grouped.setdefault(k, []).append(i)
                        pred_means = []
                        tgt_means = []
                        for idxs in grouped.values():
                            pred_means.append(preds[idxs].mean())
                            tgt_means.append(targets[idxs].mean())
                        preds = torch.stack(pred_means, dim=0)
                        targets = torch.stack(tgt_means, dim=0)
                    if args.label_noise_std > 0:
                        targets = targets + torch.randn_like(targets) * args.label_noise_std
                    weights = torch.ones_like(targets)
                    raw_loss = self.loss_fn(preds, targets)
                    raw_loss = self._apply_skew(raw_loss, preds, targets, args)
                    loss = torch.mean(raw_loss * weights)
                    if distill_loss is not None:
                        loss = loss + float(self.distill_alpha) * distill_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1
        return total_loss / max(1, steps)

    @torch.no_grad()
    def evaluate(self, loader, args=None) -> float:
        self.model.eval()
        loss_fn = nn.SmoothL1Loss(beta=1.0, reduction="mean")
        total_loss = 0.0
        steps = 0
        for batch in loader:
            if batch is None:
                continue
            if args and getattr(args, "mil_attention", False):
                if self._heteroscedastic_enabled(args):
                    preds_z, _, _, _ = self._mil_predict_batch(batch, return_logvar=True)
                    preds = self._hetero_z_to_age(preds_z, args)
                else:
                    preds, _, _ = self._mil_predict_batch(batch)
                targets = batch["age_days"].to(self.device, non_blocking=True).view(-1)
                raw_loss = loss_fn(preds, targets)
                raw_loss = self._apply_skew(raw_loss, preds, targets, args)
                loss = torch.mean(raw_loss)
                total_loss += float(loss.item())
                steps += 1
                continue
            imgs = batch["image"].to(self.device, non_blocking=True)
            targets = batch["age_days"].to(self.device, non_blocking=True)
            days = batch["day"].to(self.device, non_blocking=True)

            if args and getattr(args, "early_fusion", False):
                keys = self._group_keys(batch, batch["day"], aggregate_by_rat=getattr(args, "aggregate_by_rat", False))
                grouped = {}
                for i, k in enumerate(keys):
                    grouped.setdefault(k, []).append(i)
                fused_imgs = []
                fused_targets = []
                fused_days = []
                for idxs in grouped.values():
                    fused_imgs.append(imgs[idxs].mean(dim=0, keepdim=False))
                    fused_targets.append(targets[idxs].mean())
                    fused_days.append(days[idxs].mean())
                imgs = torch.stack(fused_imgs, dim=0)
                targets = torch.stack(fused_targets, dim=0)
                days = torch.stack(fused_days, dim=0)

            if args and args.aggregate_features and not getattr(args, "early_fusion", False):
                feats = self.model.extract_spatial_features(imgs)
                keys = self._group_keys(batch, batch["day"], aggregate_by_rat=getattr(args, "aggregate_by_rat", False))
                grouped = {}
                for i, k in enumerate(keys):
                    grouped.setdefault(k, []).append(i)
                feat_means = []
                tgt_means = []
                day_means = []
                for idxs in grouped.values():
                    feat_means.append(feats[idxs].mean(dim=0, keepdim=False))
                    tgt_means.append(targets[idxs].mean())
                    day_means.append(days[idxs].mean())
                feats_cat = torch.stack(feat_means, dim=0)
                targets = torch.stack(tgt_means, dim=0)
                days_group = torch.stack(day_means, dim=0)
                preds, _ = self.model.head(feats_cat)
                preds = preds.view(-1)
            else:
                preds, _ = self.model(imgs)
                preds = preds.view(-1)
                targets = targets.view(-1)
                if args and getattr(args, "late_fusion", False):
                    keys = self._group_keys(batch, batch["day"], aggregate_by_rat=getattr(args, "aggregate_by_rat", False))
                    grouped = {}
                    for i, k in enumerate(keys):
                        grouped.setdefault(k, []).append(i)
                    pred_means = []
                    tgt_means = []
                    for idxs in grouped.values():
                        pred_means.append(preds[idxs].mean())
                        tgt_means.append(targets[idxs].mean())
                    preds = torch.stack(pred_means, dim=0)
                    targets = torch.stack(tgt_means, dim=0)
            weights = torch.ones_like(targets)
            raw_loss = loss_fn(preds, targets)
            raw_loss = self._apply_skew(raw_loss, preds, targets, args)
            loss = torch.mean(raw_loss * weights)
            total_loss += float(loss.item())
            steps += 1
        return total_loss / max(1, steps)

    @torch.no_grad()
    def predict_to_csv(self, loader, output_name: str, args, device, correction: Optional[Tuple[str, object]] = None, save_saliency_dir=None):
        """Run inference and save per-rat/day detailed results to CSV."""
        if loader is None:
            return
        if save_saliency_dir:
            save_saliency_dir.mkdir(parents=True, exist_ok=True)
        import numpy as np  # ensure available in local scope

        rows = []
        self.model.eval()
        for batch in loader:
            if batch is None:
                continue
            if getattr(args, "mil_attention", False):
                use_hetero = self._heteroscedastic_enabled(args)
                if use_hetero:
                    preds_z_t, _, _, pred_log_var_z_t = self._mil_predict_batch(batch, return_logvar=True)
                    preds_t = self._hetero_z_to_age(preds_z_t, args)
                    pred_log_var_t = self._hetero_logvar_z_to_age(pred_log_var_z_t, args) if pred_log_var_z_t is not None else None
                else:
                    preds_t, _, _ = self._mil_predict_batch(batch)
                    pred_log_var_t = None
                preds = preds_t.detach().cpu().view(-1).numpy()
                pred_log_var_np = pred_log_var_t.detach().cpu().view(-1).numpy() if pred_log_var_t is not None else None
                pred_sigma_np = np.exp(0.5 * pred_log_var_np) if pred_log_var_np is not None else None
                targets_np = batch["age_days"].detach().cpu().view(-1).numpy()
                days_np = batch["day"].detach().cpu().view(-1).numpy()
                bag_sizes_kept = batch.get("bag_sizes")
                bag_sizes_raw = batch.get("bag_sizes_raw")
                bag_qc_dropped = batch.get("bag_qc_dropped")
                kept_np = bag_sizes_kept.detach().cpu().view(-1).numpy() if torch.is_tensor(bag_sizes_kept) else np.full(len(preds), np.nan)
                raw_np = bag_sizes_raw.detach().cpu().view(-1).numpy() if torch.is_tensor(bag_sizes_raw) else kept_np.copy()
                dropped_np = bag_qc_dropped.detach().cpu().view(-1).numpy() if torch.is_tensor(bag_qc_dropped) else np.zeros(len(preds), dtype=float)
                lowconf_thr = int(max(0, getattr(args, "mil_infer_lowconf_bag_size", 0) or 0))
                groups = list(batch.get("group", ["Unknown"] * len(preds)))
                rats = list(batch.get("rat_id", [""] * len(preds)))
                eyes = list(batch.get("eye", ["Unknown"] * len(preds)))
                sexes = list(batch.get("sex", ["Unknown"] * len(preds)))
                cohorts = list(batch.get("cohort", ["Unknown"] * len(preds)))

                if correction is not None:
                    mode, params = correction
                    coh_arr = np.array(cohorts).astype(str)
                    if mode in {"poly_cohort", "linear_cohort"}:
                        young_mask = np.isin(coh_arr, ["1", "2"])
                        old_mask = coh_arr == "3"
                        if young_mask.any() and "young" in params:
                            if mode == "poly_cohort":
                                preds[young_mask] = apply_poly_correction(preds[young_mask], params["young"])
                            else:
                                alpha, beta = params["young"]
                                preds[young_mask] = apply_correction(targets_np[young_mask], preds[young_mask], alpha, beta)
                        if old_mask.any() and "old" in params:
                            if mode == "poly_cohort":
                                preds[old_mask] = apply_poly_correction(preds[old_mask], params["old"])
                            else:
                                alpha, beta = params["old"]
                                preds[old_mask] = apply_correction(targets_np[old_mask], preds[old_mask], alpha, beta)
                    elif mode in {"poly_cohort_exact", "linear_cohort_exact"}:
                        for c, coeffs in params.items():
                            mask = coh_arr == str(c)
                            if not mask.any():
                                continue
                            if mode == "poly_cohort_exact":
                                preds[mask] = apply_poly_correction(preds[mask], coeffs)
                            else:
                                alpha, beta = coeffs
                                preds[mask] = apply_correction(targets_np[mask], preds[mask], alpha, beta)
                    else:
                        if mode == "poly":
                            preds = apply_poly_correction(preds, params)
                        else:
                            alpha, beta = params
                            preds = apply_correction(targets_np, preds, alpha, beta)

                for j, (rat, eye, sex, coh, grp, d, y_true, y_pred, n_kept, n_raw, n_drop) in enumerate(zip(
                    rats, eyes, sexes, cohorts, groups, days_np, targets_np, preds, kept_np, raw_np, dropped_np
                )):
                    n_kept_i = int(n_kept) if np.isfinite(n_kept) else None
                    n_raw_i = int(n_raw) if np.isfinite(n_raw) else None
                    n_drop_i = int(n_drop) if np.isfinite(n_drop) else None
                    row = {
                        "rat_id": rat,
                        "eye": eye,
                        "sex": sex,
                        "cohort": coh,
                        "group": grp,
                        "day": float(d),
                        "age_true": float(y_true),
                        "age_pred": float(y_pred),
                        "mil_bag_n_kept": n_kept_i,
                        "mil_bag_n_raw": n_raw_i,
                        "mil_bag_n_qc_dropped": n_drop_i,
                        "mil_low_conf_bag": bool(lowconf_thr > 0 and n_kept_i is not None and n_kept_i < lowconf_thr),
                    }
                    if pred_log_var_np is not None:
                        # sigma is a learned uncertainty estimate from the heteroscedastic head.
                        row["pred_log_var"] = float(pred_log_var_np[j])
                        row["pred_sigma"] = float(pred_sigma_np[j])
                    rows.append(row)
                continue
            imgs = batch["image"].to(device, non_blocking=True)
            targets = batch["age_days"].to(device, non_blocking=True)

            if getattr(args, "early_fusion", False):
                eyes_list = [str(e) for e in batch["eye"]]
                keys = [(r, e, float(d.item())) for r, e, d in zip(batch["rat_id"], eyes_list, batch["day"])]
                grouped = {}
                for i, k in enumerate(keys):
                    grouped.setdefault(k, []).append(i)
                fused_imgs = []
                fused_targets = []
                fused_days = []
                meta = []
                for k, idxs in grouped.items():
                    if len(k) == 3:
                        rat_k, eye_k, day_k = k
                    else:
                        rat_k, day_k = k
                        eye_k = "both"
                    fused_imgs.append(imgs[idxs].mean(dim=0, keepdim=False))
                    fused_targets.append(targets[idxs].mean())
                    fused_days.append(batch["day"][idxs].mean())
                    meta.append({
                        "rat_id": rat_k,
                        "eye": eye_k,
                        "day": float(day_k),
                        "group": batch["group"][idxs[0]],
                        "sex": batch["sex"][idxs[0]],
                        "cohort": batch["cohort"][idxs[0]],
                    })
                imgs = torch.stack(fused_imgs, dim=0)
                targets = torch.stack(fused_targets, dim=0)
                days = torch.stack(fused_days, dim=0)

            if args.aggregate_features and not getattr(args, "early_fusion", False):
                feats = self.model.extract_spatial_features(imgs)
                eyes_list = [str(e) for e in batch["eye"]]
                keys = [(r, e, float(d.item())) for r, e, d in zip(batch["rat_id"], eyes_list, batch["day"])]
                grouped = {}
                for i, k in enumerate(keys):
                    grouped.setdefault(k, []).append(i)
                feat_means = []
                tgt_means = []
                day_means = []
                meta = []
                for k, idxs in grouped.items():
                    if len(k) == 3:
                        rat_k, eye_k, day_k = k
                    else:
                        rat_k, day_k = k
                        eye_k = "both"
                    feat_means.append(feats[idxs].mean(dim=0, keepdim=False))
                    tgt_means.append(targets[idxs].mean())
                    day_means.append(batch["day"][idxs].mean())
                    meta.append({
                        "rat_id": rat_k,
                        "eye": eye_k,
                        "day": float(day_k),
                        "group": batch["group"][idxs[0]],
                        "sex": batch["sex"][idxs[0]],
                        "cohort": batch["cohort"][idxs[0]],
                    })
                feats_cat = torch.stack(feat_means, dim=0)
                preds, _ = self.model.head(feats_cat)
                preds = preds.view(-1)
                preds = preds.detach().cpu().view(-1).numpy()
                targets = torch.stack(tgt_means, dim=0).detach().cpu().view(-1).numpy()
                days = torch.stack(day_means, dim=0).detach().cpu().view(-1).numpy()
                metas = meta
            else:
                preds_orig, _ = self.model(imgs)
                preds_list = [preds_orig]
                if getattr(args, "tta", False):
                    flipped = torch.flip(imgs, dims=[3])  # flip width dim
                    preds_flip, _ = self.model(flipped)
                    preds_list.append(preds_flip)
                preds = torch.stack(preds_list).mean(dim=0)
                if save_saliency_dir and hasattr(self.model, "get_age_saliency_maps"):
                    # Saliency on original (non-flipped) images
                    if hasattr(self.model, "keep_spatial_tokens") and not bool(getattr(self.model, "keep_spatial_tokens")):
                        if not getattr(self, "_warned_nonspatial_saliency", False):
                            print("[SAL] Skipping saliency export: model is in CLS-only mode (use --keep-spatial-tokens for spatial maps).")
                            self._warned_nonspatial_saliency = True
                    else:
                        try:
                            import numpy as np
                            from matplotlib import cm
                            try:
                                from scipy.ndimage import gaussian_filter
                            except Exception:
                                gaussian_filter = None

                            sal = self.model.get_age_saliency_maps(imgs)
                            sal = sal.detach().cpu().numpy()
                            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
                            std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
                            for i, (rat, eye, day) in enumerate(zip(batch["rat_id"], batch.get("eye", ["Unknown"]*len(imgs)), batch["day"])):
                                fname = f"{rat}_{eye}_{float(day):.1f}_{i}.png"
                                arr = sal[i, 0] if sal.ndim == 4 else sal[i]
                                # percentile scaling to reduce outlier influence
                                p2, p98 = np.percentile(arr, [2, 98])
                                arr = (arr - p2) / (p98 - p2 + 1e-6)
                                arr = np.clip(arr, 0, 1)
                                if gaussian_filter is not None:
                                    arr = gaussian_filter(arr, sigma=1.0)
                                # recover RGB image for overlay
                                base = imgs[i].detach().cpu().permute(1, 2, 0).numpy()
                                base = np.clip((base * std + mean), 0, 1)
                                overlay = base.copy()
                                # highlight top 5% pixels in red
                                mask = arr >= np.percentile(arr, 95)
                                if mask.any():
                                    m = np.expand_dims(mask.astype(float), axis=2)
                                    overlay = np.clip(overlay * (1 - 0.5 * m) + m * np.array([1.0, 0.0, 0.0]), 0, 1)
                                from PIL import Image  # lazy import
                                im = Image.fromarray((overlay * 255).astype("uint8"), mode="RGB")
                                im.save(save_saliency_dir / fname)
                        except Exception as e:
                            print(f"[SAL] Failed to save saliency for batch (skipping): {e}")
                if getattr(args, "late_fusion", False):
                    preds = preds.view(-1)
                    targets = targets.view(-1)
                    keys = self._group_keys(batch, batch["day"], aggregate_by_rat=getattr(args, "aggregate_by_rat", False))
                    grouped = {}
                    for i, k in enumerate(keys):
                        grouped.setdefault(k, []).append(i)
                    pred_means = []
                    tgt_means = []
                    day_means = []
                    meta = []
                    for k, idxs in grouped.items():
                        if len(k) == 3:
                            rat_k, eye_k, day_k = k
                        else:
                            rat_k, day_k = k
                            eye_k = "both"
                        pred_means.append(preds[idxs].mean())
                        tgt_means.append(targets[idxs].mean())
                        day_means.append(batch["day"][idxs].mean())
                        meta.append({
                            "rat_id": rat_k,
                            "eye": eye_k,
                            "day": float(day_k),
                            "group": batch["group"][idxs[0]],
                            "sex": batch["sex"][idxs[0]],
                            "cohort": batch["cohort"][idxs[0]],
                        })
                    preds = torch.stack(pred_means, dim=0).detach().cpu().view(-1).numpy()
                    targets = torch.stack(tgt_means, dim=0).detach().cpu().view(-1).numpy()
                    days = torch.stack(day_means, dim=0).detach().cpu().view(-1).numpy()
                    metas = meta
                else:
                    preds = preds.detach().cpu().view(-1).numpy()
                    targets = targets.detach().cpu().view(-1).numpy()
                    days = batch["day"].detach().cpu().view(-1).numpy()
                    metas = None
            # Prepare meta fields
            if metas is None:
                groups = batch["group"]
                rats = batch["rat_id"]
                if getattr(args, "aggregate_by_rat", False):
                    eyes = ["both"] * len(rats)
                else:
                    eyes = batch.get("eye", ["Unknown"] * len(rats)) if isinstance(batch, dict) else ["Unknown"] * len(rats)
                sexes = batch.get("sex", ["Unknown"] * len(rats)) if isinstance(batch, dict) else ["Unknown"] * len(rats)
                cohorts = batch.get("cohort", ["Unknown"] * len(rats)) if isinstance(batch, dict) else ["Unknown"] * len(rats)
                labels_for_corr = list(groups)
            else:
                groups = [m["group"] for m in metas]
                rats = [m["rat_id"] for m in metas]
                eyes = [m["eye"] for m in metas]
                sexes = [m["sex"] for m in metas]
                cohorts = [m["cohort"] for m in metas]
                labels_for_corr = groups

            if correction is not None:
                mode, params = correction
                coh_arr = np.array(cohorts).astype(str)
                if mode in {"poly_cohort", "linear_cohort"}:
                    young_mask = np.isin(coh_arr, ["1", "2"])
                    old_mask = coh_arr == "3"
                    if young_mask.any() and "young" in params:
                        if mode == "poly_cohort":
                            preds[young_mask] = apply_poly_correction(preds[young_mask], params["young"])
                        else:
                            alpha, beta = params["young"]
                            preds[young_mask] = apply_correction(targets[young_mask], preds[young_mask], alpha, beta)
                    if old_mask.any() and "old" in params:
                        if mode == "poly_cohort":
                            preds[old_mask] = apply_poly_correction(preds[old_mask], params["old"])
                        else:
                            alpha, beta = params["old"]
                            preds[old_mask] = apply_correction(targets[old_mask], preds[old_mask], alpha, beta)
                elif mode in {"poly_cohort_exact", "linear_cohort_exact"}:
                    for c, coeffs in params.items():
                        mask = coh_arr == str(c)
                        if not mask.any():
                            continue
                        if mode == "poly_cohort_exact":
                            preds[mask] = apply_poly_correction(preds[mask], coeffs)
                        else:
                            alpha, beta = coeffs
                            preds[mask] = apply_correction(targets[mask], preds[mask], alpha, beta)
                else:
                    # fallback global correction
                    if mode == "poly":
                        preds = apply_poly_correction(preds, params)
                    else:
                        alpha, beta = params
                        preds = apply_correction(targets, preds, alpha, beta)
            if metas is None:
                for rat, eye, sex, coh, grp, d, y_true, y_pred in zip(rats, eyes, sexes, cohorts, groups, days, targets, preds):
                    rows.append({
                        "rat_id": rat,
                        "eye": eye,
                        "sex": sex,
                        "cohort": coh,
                        "group": grp,
                        "day": float(d),
                        "age_true": float(y_true),
                        "age_pred": float(y_pred),
                    })
            else:
                for m, y_true, y_pred in zip(metas, targets, preds):
                    rows.append({
                        "rat_id": m["rat_id"],
                        "eye": m["eye"],
                        "sex": m["sex"],
                        "cohort": m["cohort"],
                        "group": m["group"],
                        "day": float(m["day"]),
                        "age_true": float(y_true),
                        "age_pred": float(y_pred),
                    })

        df_pred = pd.DataFrame(rows)
        if df_pred.empty:
            return

        if getattr(args, "no_aggregate", False):
            # Keep per-image rows
            df_agg = df_pred.copy()
            df_agg["RAG"] = df_agg["age_pred"] - df_agg["age_true"]
        else:
            # Aggregate per rat/eye/day (keep eyes separate; average slices per eye/day)
            agg_cols = {
                "age_true": "mean",
                "age_pred": "mean",
                "group": "first",
                "eye": "first",
                "sex": "first",
                "cohort": "first",
            }
            if "mil_bag_n_kept" in df_pred.columns:
                agg_cols["mil_bag_n_kept"] = "mean"
            if "mil_bag_n_raw" in df_pred.columns:
                agg_cols["mil_bag_n_raw"] = "mean"
            if "mil_bag_n_qc_dropped" in df_pred.columns:
                agg_cols["mil_bag_n_qc_dropped"] = "mean"
            if "mil_low_conf_bag" in df_pred.columns:
                agg_cols["mil_low_conf_bag"] = "max"
            if "pred_log_var" in df_pred.columns:
                agg_cols["pred_log_var"] = "mean"
            if "pred_sigma" in df_pred.columns:
                agg_cols["pred_sigma"] = "mean"
            df_agg = df_pred.groupby(["rat_id", "eye", "day"], as_index=False).agg(agg_cols)
            for c in ("mil_bag_n_kept", "mil_bag_n_raw", "mil_bag_n_qc_dropped"):
                if c in df_agg.columns:
                    df_agg[c] = np.rint(pd.to_numeric(df_agg[c], errors="coerce")).astype("Int64")
            if "mil_low_conf_bag" in df_agg.columns:
                df_agg["mil_low_conf_bag"] = df_agg["mil_low_conf_bag"].astype(bool)
            df_agg["RAG"] = df_agg["age_pred"] - df_agg["age_true"]

        meta_df = load_metadata(
            csv_path=args.csv,
            image_types=IMAGE_TYPES,
            day_whitelist=getattr(args, "day_whitelist", DAY_WHITELIST),
            include_recovery_days=True,
            cohorts_to_keep=COHORTS_TO_KEEP,
            exclude_recovery_paths=False,
            verbose=False,
        )
        # Backfill missing/unknown cohort labels conservatively.
        # Do not overwrite existing cohort labels in predictions because `rat_id`
        # can be reused across cohorts in this dataset.
        try:
            rat_cohort_unique = (
                meta_df[["rat_id", "cohort"]]
                .dropna()
                .astype({"rat_id": str, "cohort": str})
                .groupby("rat_id")["cohort"]
                .agg(lambda s: s.iloc[0] if s.nunique() == 1 else np.nan)
            )
            mapped_cohort = df_agg["rat_id"].astype(str).map(rat_cohort_unique)
            if "cohort" not in df_agg.columns:
                df_agg["cohort"] = mapped_cohort
            else:
                cohort_existing = df_agg["cohort"]
                missing_mask = cohort_existing.isna() | cohort_existing.astype(str).str.strip().isin(["", "Unknown", "nan"])
                if missing_mask.any():
                    df_agg.loc[missing_mask, "cohort"] = mapped_cohort.loc[missing_mask].fillna(df_agg.loc[missing_mask, "cohort"])
        except Exception:
            pass

        out_path = args.pred_csv.parent / output_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_agg.to_csv(out_path, index=False)
        print(f"[PRED] Saved detailed results to: {out_path} (N={len(df_agg)})")
        if "mil_low_conf_bag" in df_agg.columns:
            try:
                n_lowconf = int(df_agg["mil_low_conf_bag"].astype(bool).sum())
                if n_lowconf > 0:
                    print(f"[PRED][MIL-QC] Low-confidence bags flagged: {n_lowconf}/{len(df_agg)}")
            except Exception:
                pass
        try:
            summary = df_agg.groupby(['cohort','group','day'])['RAG'].mean().reset_index()
            print(summary.to_string(index=False))
        except Exception:
            pass

        try:
            from scipy.stats import pearsonr, spearmanr  # lazy import

            pearson_r, pearson_p = pearsonr(df_agg["age_true"], df_agg["age_pred"])
            spearman_r, spearman_p = spearmanr(df_agg["age_true"], df_agg["age_pred"])
        except Exception:
            pearson_r = spearman_r = float("nan")
            pearson_p = spearman_p = float("nan")
        ss_res = float(np.sum((df_agg["age_true"] - df_agg["age_pred"]) ** 2))
        ss_tot = float(np.sum((df_agg["age_true"] - df_agg["age_true"].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mae = float(np.mean(np.abs(df_agg["age_true"] - df_agg["age_pred"])))
        rmse = float(np.sqrt(np.mean((df_agg["age_true"] - df_agg["age_pred"]) ** 2)))
        print(
            f"[PRED] MAE={mae:.2f} | RMSE={rmse:.2f} | "
            f"Pearson r={pearson_r:.4f} (p={pearson_p:.3g}) | "
            f"Spearman ρ={spearman_r:.4f} (p={spearman_p:.3g}) | "
            f"R²={r2:.4f}"
        )

    @staticmethod
    @torch.no_grad()
    def collect_preds(model, loader, device):
        ys_true, ys_pred, ys_coh = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                if isinstance(batch, dict) and "bags" in batch and hasattr(model, "mil_head") and getattr(model, "mil_head", None) is not None:
                    bags = batch.get("bags", [])
                    bag_sizes = [int(b.shape[0]) for b in bags]
                    all_imgs = torch.cat([b.to(device, non_blocking=True) for b in bags], dim=0)
                    feats = model.extract_image_features(all_imgs)
                    feat_splits = torch.split(feats, bag_sizes, dim=0)
                    pred_list = []
                    for f in feat_splits:
                        p, _ = model.mil_predict_from_features(f)
                        pred_list.append(p.view(-1)[0])
                    preds = torch.stack(pred_list, dim=0)
                    if bool(getattr(model, "heteroscedastic_regression", False)):
                        mean = float(getattr(model, "hetero_target_mean", 0.0) or 0.0)
                        std = float(getattr(model, "hetero_target_std", 1.0) or 1.0)
                        if np.isfinite(std) and std > 1e-6:
                            preds = preds * std + mean
                    targets = batch["age_days"].to(device, non_blocking=True)
                    ys_true.append(targets.detach().cpu().view(-1).numpy())
                    ys_pred.append(preds.detach().cpu().view(-1).numpy())
                    coh = batch.get("cohort", ["Unknown"] * len(preds))
                    coh = np.array(list(coh)).reshape(-1)
                else:
                    imgs = batch["image"].to(device, non_blocking=True)
                    targets = batch["age_days"].to(device, non_blocking=True)
                    preds, _ = model(imgs)
                    ys_true.append(targets.detach().cpu().view(-1).numpy())
                    ys_pred.append(preds.detach().cpu().view(-1).numpy())
                    coh = batch.get("cohort", ["Unknown"] * len(preds))
                    coh = np.array(list(coh)).reshape(-1)
                ys_coh.append(coh)
        if not ys_true:
            return None, None, None
        return np.concatenate(ys_true), np.concatenate(ys_pred), np.concatenate(ys_coh)
