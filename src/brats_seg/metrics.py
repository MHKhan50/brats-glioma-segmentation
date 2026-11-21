"""Dice and 95th-percentile Hausdorff distance per region, with the BraTS conventions for empty masks."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

EMPTY_HD95 = 373.13  # BraTS convention when exactly one of prediction and reference is empty


def dice(pred: np.ndarray, ref: np.ndarray) -> float:
    p, r = pred.astype(bool), ref.astype(bool)
    if not p.any() and not r.any():
        return 1.0
    return float(2 * (p & r).sum() / (p.sum() + r.sum()))


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3)), border_value=0)


def hd95(pred: np.ndarray, ref: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    p, r = pred.astype(bool), ref.astype(bool)
    if not p.any() and not r.any():
        return 0.0
    if not p.any() or not r.any():
        return EMPTY_HD95
    sp, sr = _surface(p), _surface(r)
    dt_r = ndimage.distance_transform_edt(~sr, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    d = np.concatenate([dt_r[sp], dt_p[sr]])
    return float(np.percentile(d, 95))


def ece(prob: np.ndarray, target: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error of voxel probabilities (flattened)."""
    p, y = prob.ravel().astype(np.float64), target.ravel().astype(np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    out = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)
