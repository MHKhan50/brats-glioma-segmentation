import numpy as np
import pandas as pd

from brats_seg.postprocess import apply_threshold, choose_threshold, summarise


def _df():
    # case 0: small false-positive ET (ref empty) -> removing helps; case 1: real ET, large prediction -> keep
    rows = [
        {"model": "unet", "split": s, "case_id": f"{s}{i}", "patient": f"{s}{i}", "et_pred_voxels": v, "et_ref_voxels": r,
         "dice_kept": dk, "hd95_kept": hk, "dice_removed": dr, "hd95_removed": hr}
        for s in ("val", "test")
        for i, (v, r, dk, hk, dr, hr) in enumerate([(40, 0, 0.0, 373.13, 1.0, 0.0), (5000, 4000, 0.85, 3.0, 0.0, 373.13)])
    ]
    return pd.DataFrame(rows)


def test_apply_threshold():
    df = _df()[lambda d: d.split == "val"].reset_index(drop=True)
    r = apply_threshold(df, 50)
    assert list(r["removed"]) == [True, False]
    assert np.isclose(r["dice"].mean(), (1.0 + 0.85) / 2)


def test_choose_threshold_prefers_smallest_best():
    t, curve = choose_threshold(_df()[lambda d: d.split == "val"])
    assert t == 50 and len(curve) > 3


def test_summarise_counts():
    s = summarise(_df())["unet"]
    assert s["chosen_threshold"] == 50
    assert s["test_false_positive_et_before"] == 1 and s["test_false_positive_et_after"] == 0
    assert s["test_true_et_removed"] == 0
    assert s["test_et_dice_after"] > s["test_et_dice_before"]
