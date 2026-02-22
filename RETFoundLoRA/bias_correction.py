"""
Bias correction utilities (de Lange et al. 2020 method).
Models prediction as a function of true age and subtracts that trend so RAG is age-independent.
"""

import numpy as np
from typing import Tuple, Sequence


def fit_linear_correction(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Fit bias trend: y_pred ≈ alpha * y_true + beta.
    (Prediction regressed on true age, per de Lange et al. 2020.)
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size != y_pred.size or y_true.size == 0:
        return 1.0, 0.0
    A = np.vstack([y_true, np.ones_like(y_true)]).T
    coeffs, *_ = np.linalg.lstsq(A, y_pred, rcond=None)
    alpha, beta = coeffs.tolist()
    return float(alpha), float(beta)


def apply_correction(y_true: np.ndarray, y_pred: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Apply de Lange correction:
      expected_pred = alpha * y_true + beta
      corrected = y_pred + (y_true - expected_pred)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    expected_pred = (alpha * y_true) + beta
    return y_pred + (y_true - expected_pred)


def fit_poly_correction(y_true: np.ndarray, y_pred: np.ndarray, degree: int = 2) -> Sequence[float]:
    """Fit polynomial coeffs c0..cn such that y_true ≈ poly(y_pred)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size != y_pred.size or y_true.size == 0:
        return [1.0, 0.0]
    coeffs = np.polyfit(y_pred, y_true, degree)
    return coeffs  # highest power first


def apply_poly_correction(y_pred: np.ndarray, coeffs: Sequence[float]) -> np.ndarray:
    poly = np.poly1d(coeffs)
    return poly(np.asarray(y_pred, dtype=float))
