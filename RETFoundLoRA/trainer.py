"""Training and evaluation engine for RETFound LoRA age regression."""

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

    def _mil_predict_batch(self, batch):
        """MIL forward for a collated bag batch (bag = rat_id/eye/day)."""
        bags = batch.get("bags", [])
        if not bags:
            return torch.empty(0, device=self.device), []
        bag_sizes = [int(b.shape[0]) for b in bags]
        all_imgs = torch.cat([b.to(self.device, non_blocking=True) for b in bags], dim=0)
        feats = self.model.extract_image_features(all_imgs)
        feat_splits = torch.split(feats, bag_sizes, dim=0)
        preds = []
        attn_weights = []
        for f in feat_splits:
            p, w = self.model.mil_predict_from_features(f)
            preds.append(p.view(-1)[0])
            attn_weights.append(w)
        return torch.stack(preds, dim=0), attn_weights

    def train_one_epoch(self, loader, optimizer, args) -> float:
        self.model.train()
        total_loss = 0.0
        steps = 0
        for batch in loader:
            if batch is None:
                continue
            if getattr(args, "mil_attention", False):
                preds, _ = self._mil_predict_batch(batch)
                targets = batch["age_days"].to(self.device, non_blocking=True).view(-1)
                if args.label_noise_std > 0:
                    targets = targets + torch.randn_like(targets) * args.label_noise_std
                raw_loss = self.loss_fn(preds, targets)
                raw_loss = self._apply_skew(raw_loss, preds, targets, args)
                loss = torch.mean(raw_loss)

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
                preds, _ = self._mil_predict_batch(batch)
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
                preds_t, _ = self._mil_predict_batch(batch)
                preds = preds_t.detach().cpu().view(-1).numpy()
                targets_np = batch["age_days"].detach().cpu().view(-1).numpy()
                days_np = batch["day"].detach().cpu().view(-1).numpy()
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

                for rat, eye, sex, coh, grp, d, y_true, y_pred in zip(rats, eyes, sexes, cohorts, groups, days_np, targets_np, preds):
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
            df_agg = df_pred.groupby(["rat_id", "eye", "day"], as_index=False).agg(agg_cols)
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
        rat_to_cohort = dict(zip(meta_df["rat_id"], meta_df["cohort"]))
        df_agg["cohort"] = df_agg["rat_id"].map(rat_to_cohort).fillna(df_agg["cohort"])

        out_path = args.pred_csv.parent / output_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_agg.to_csv(out_path, index=False)
        print(f"[PRED] Saved detailed results to: {out_path} (N={len(df_agg)})")
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
