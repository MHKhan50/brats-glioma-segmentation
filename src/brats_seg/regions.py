"""BraTS-GLI labels and the three evaluated regions.

Labels: 0 background, 1 non-enhancing tumour core (NETC), 2 surrounding FLAIR hyperintensity (SNFH, oedema),
3 enhancing tumour (ET). Regions: whole tumour WT = {1, 2, 3}, tumour core TC = {1, 3}, enhancing tumour ET = {3}.
"""
from __future__ import annotations

import numpy as np

REGIONS = ("WT", "TC", "ET")


def labels_to_regions(label: np.ndarray) -> np.ndarray:
    """(D, H, W) integer labels -> (3, D, H, W) uint8 masks for WT, TC, ET."""
    wt = np.isin(label, (1, 2, 3))
    tc = np.isin(label, (1, 3))
    et = label == 3
    return np.stack([wt, tc, et]).astype(np.uint8)


def enforce_nesting(masks: np.ndarray) -> np.ndarray:
    """Make ET inside TC inside WT, as in the label definition."""
    wt, tc, et = (masks[i].astype(bool) for i in range(3))
    tc = tc & wt
    et = et & tc
    return np.stack([wt, tc, et]).astype(np.uint8)


def regions_to_labels(masks: np.ndarray) -> np.ndarray:
    """(3, ...) nested WT/TC/ET masks -> labels 0-3 (for display and export)."""
    wt, tc, et = (masks[i].astype(bool) for i in range(3))
    out = np.zeros(wt.shape, np.uint8)
    out[wt] = 2
    out[tc] = 1
    out[et] = 3
    return out
