import numpy as np
import pandas as pd

from brats_seg.data import brain_bbox, normalise_to_uint8, patient_split, preprocess_arrays, uint8_to_float
from brats_seg.metrics import EMPTY_HD95, dice, ece, hd95
from brats_seg.regions import enforce_nesting, labels_to_regions, regions_to_labels


def test_regions_roundtrip_and_nesting():
    lab = np.array([0, 1, 2, 3])
    reg = labels_to_regions(lab)
    assert reg.tolist() == [[0, 1, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1]]
    assert (regions_to_labels(reg) == lab).all()
    bad = np.array([[0, 0], [1, 0], [1, 1]])  # ET and TC outside WT
    assert enforce_nesting(bad).sum() == 0


def test_dice_and_hd95_conventions():
    a = np.zeros((10, 10, 10), bool); a[2:5, 2:5, 2:5] = True
    assert dice(a, a) == 1.0 and hd95(a, a) == 0.0
    e = np.zeros_like(a)
    assert dice(e, e) == 1.0 and hd95(e, e) == 0.0
    assert dice(a, e) == 0.0 and hd95(a, e) == EMPTY_HD95
    b = np.roll(a, 2, axis=0)
    assert 0 < dice(a, b) < 1 and abs(hd95(a, b) - 2.0) < 1e-6


def test_ece_perfect_and_bad():
    y = np.array([0, 0, 1, 1])
    assert ece(y.astype(float), y) == 0.0
    assert ece(np.full(4, 0.9), y) > 0.3


def test_patient_split_keeps_timepoints_together():
    df = pd.DataFrame({"patient": [f"{i // 2:05d}" for i in range(200)]})
    s = patient_split(df)
    assert set(s) == {"train", "val", "test"}
    assert (df.assign(s=s).groupby("patient")["s"].nunique() == 1).all()


def test_preprocess_crops_resamples_and_normalises():
    rng = np.random.default_rng(0)
    vols = np.zeros((4, 40, 50, 30), np.float32)
    vols[:, 10:30, 10:40, 5:25] = rng.normal(100, 20, (4, 20, 30, 20))
    seg = np.zeros((40, 50, 30), np.uint8); seg[15:20, 15:20, 10:15] = 3; seg[14:21, 14:21, 9:16][seg[14:21, 14:21, 9:16] == 0] = 2
    img, lab, meta = preprocess_arrays(vols, seg, size=16)
    assert img.shape == (4, 16, 16, 16) and img.dtype == np.uint8 and lab.shape == (16, 16, 16)
    assert set(np.unique(lab)) <= {0, 2, 3} and 3 in lab
    assert meta["bbox"][0] == (8, 32) and meta["shape"] == [40, 50, 30]
    z = uint8_to_float(img[0]); inside = img[0] > 0
    assert abs(z[inside].mean()) < 0.2 and abs(z[inside].std() - 1) < 0.2


def test_uint8_roundtrip_precision():
    v = np.linspace(-3, 5, 1000).astype(np.float32).reshape(10, 10, 10)
    mask = np.ones_like(v, bool)
    q = normalise_to_uint8((v - v.mean()) / v.std() * 1.0, mask)
    assert q.min() >= 1 and q.max() <= 255
    assert brain_bbox(np.ones((1, 4, 4, 4)), margin=0) == [(0, 4), (0, 4), (0, 4)]
