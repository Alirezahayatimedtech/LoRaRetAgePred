"""Helper utilities for RETFound LoRA pipelines."""

from typing import Union
import numpy as np
import torch
import re


def to_rgb3(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Duplicate a single-channel image to 3 channels for RETFound compatibility."""
    if isinstance(x, torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        return x
    else:
        arr = np.asarray(x)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        return arr


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0 or y_true.size != y_pred.size:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0 or y_true.size != y_pred.size:
        return float("nan")
    y_true = y_true - y_true.mean()
    y_pred = y_pred - y_pred.mean()
    denom = np.sqrt(np.sum(y_true ** 2) * np.sum(y_pred ** 2))
    if denom == 0:
        return float("nan")
    return float(np.sum(y_true * y_pred) / denom)


def infer_eye_from_path(path_str: str) -> str:
    """Infer eye side from path string. Returns 'OD', 'OS', or 'Unknown'."""
    s = path_str.lower()
    if re.search(r"\b(od|right)\b", s):
        return "OD"
    if re.search(r"\b(os|left)\b", s):
        return "OS"
    return "Unknown"


def normalize_eye_side(eye_val, path_str: str = "", material_type: str = "") -> str:
    """
    Normalize eye side, preferring path/material hints over possibly mis-labeled metadata.
    Priority: path/material -> explicit eye field -> Unknown.
    """
    side = None
    # Prefer folder/filename/material cues (more reliable than mislabeled eye fields in CSV)
    if path_str:
        inferred = infer_eye_from_path(path_str)
        if inferred != "Unknown":
            side = inferred
    if side is None and material_type:
        mt = str(material_type).strip().lower()
        if "right" in mt:
            side = "OD"
        elif "left" in mt:
            side = "OS"
    # Fallback to provided eye metadata if path/material inconclusive
    if side is None and eye_val is not None:
        s = str(eye_val).strip().lower()
        if s in {"od", "r", "right"}:
            side = "OD"
        elif s in {"os", "l", "left"}:
            side = "OS"
    return side if side is not None else "Unknown"
