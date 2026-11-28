"""Enhancing-tumour post-processing, chosen on validation and applied once to test.

The rule (as in Isensee et al., 2018, nnU-Net for BraTS): if the predicted enhancing tumour (ET) has fewer than T
voxels, ET is set to empty (those voxels stay in the tumour core, so WT and TC do not change). T is chosen on the
validation cases only, separately for each model's seed ensemble, by mean ET Dice; the test set is then scored once.

Because the rule either keeps or removes the whole ET prediction, each case needs only two scores (kept, removed)
and the predicted ET volume; every threshold is then a simple lookup.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import dice, hd95
from .regions import enforce_nesting, labels_to_regions

THRESHOLDS = (0, 25, 50, 100, 200, 300, 500, 750, 1000)


def case_scores(name: str, split: str, case: dict, prob: np.ndarray, ref: np.ndarray) -> dict:
    pred = enforce_nesting(prob >= 0.5)
    sp = tuple(case.get("spacing", (1.0, 1.0, 1.0)))
    et_pred, et_ref, empty = pred[2], ref[2], np.zeros_like(pred[2])
    return {"model": name, "split": split, "case_id": case["case_id"], "patient": case["patient"],
            "et_pred_voxels": int(et_pred.sum()), "et_ref_voxels": int(et_ref.sum()),
            "dice_kept": dice(et_pred, et_ref), "hd95_kept": hd95(et_pred, et_ref, sp),
            "dice_removed": dice(empty, et_ref), "hd95_removed": hd95(empty, et_ref, sp)}


def apply_threshold(df: pd.DataFrame, t: int) -> pd.DataFrame:
    removed = df["et_pred_voxels"] < t
    return pd.DataFrame({"dice": np.where(removed, df["dice_removed"], df["dice_kept"]),
                         "hd95": np.where(removed, df["hd95_removed"], df["hd95_kept"]),
                         "removed": removed}, index=df.index)


def choose_threshold(val: pd.DataFrame, thresholds=THRESHOLDS) -> tuple[int, list[dict]]:
    curve = []
    for t in thresholds:
        r = apply_threshold(val, t)
        curve.append({"threshold": t, "val_et_dice": round(float(r["dice"].mean()), 4),
                      "val_et_hd95": round(float(r["hd95"].mean()), 2), "val_removed": int(r["removed"].sum())})
    best = max(curve, key=lambda c: (c["val_et_dice"], -c["threshold"]))  # ties: the smaller threshold
    return best["threshold"], curve


def paired_bootstrap(a: np.ndarray, b: np.ndarray, groups: np.ndarray, n: int = 2000, seed: int = 0) -> list[float]:
    """Mean of b - a with a patient-level bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    d = pd.Series(b - a).groupby(groups).mean()
    boots = [d.values[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return [round(float((b - a).mean()), 4), round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


def summarise(df: pd.DataFrame) -> dict:
    out = {"rule": "ET set to empty when the predicted ET has fewer than T voxels; T chosen on validation by mean ET Dice",
           "thresholds_tried": list(THRESHOLDS)}
    for name, g in df.groupby("model"):
        val, test = g[g.split == "val"], g[g.split == "test"].reset_index(drop=True)
        t, curve = choose_threshold(val)
        before, after = apply_threshold(test, 0), apply_threshold(test, t)
        empty_ref = test["et_ref_voxels"] == 0
        out[name] = {
            "chosen_threshold": t, "validation_curve": curve, "val_cases": len(val), "test_cases": len(test),
            "test_et_dice_before": round(float(before["dice"].mean()), 4),
            "test_et_dice_after": round(float(after["dice"].mean()), 4),
            "test_et_dice_change": paired_bootstrap(before["dice"].values, after["dice"].values, test["patient"].values),
            "test_et_hd95_before": round(float(before["hd95"].mean()), 2),
            "test_et_hd95_after": round(float(after["hd95"].mean()), 2),
            "test_cases_removed": int(after["removed"].sum()),
            "test_cases_without_et": int(empty_ref.sum()),
            "test_false_positive_et_before": int((empty_ref & (test["et_pred_voxels"] > 0)).sum()),
            "test_false_positive_et_after": int((empty_ref & (test["et_pred_voxels"] >= t) & (test["et_pred_voxels"] > 0)).sum()),
            "test_true_et_removed": int((~empty_ref & after["removed"] & (test["et_pred_voxels"] > 0)).sum()),
        }
    return out


def run(data_dir: str, run_dir: str, out_dir: str, names=("unet", "swinunetr"), seeds=(0, 1, 2), limit: int = 0) -> dict:
    import torch

    from .evaluate import to_native
    from .train import CaseDataset, build_model, predict_volume
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_cases = json.loads((Path(data_dir) / "cases.json").read_text())
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        models = []
        for s in seeds:
            ck = Path(run_dir) / f"{name}_s{s}_best.pt"
            if ck.exists():
                m = build_model(name)
                m.load_state_dict(torch.load(ck, map_location="cpu")["state_dict"])
                models.append(m.to(dev).eval())
        if not models:
            continue
        for split in ("val", "test"):
            cases = [c for c in all_cases if c["split"] == split]
            cases = cases[:limit] if limit else cases
            for i, c in enumerate(cases, 1):
                x, _ = CaseDataset([str(Path(data_dir) / "images" / f"{c['case_id']}.npy")], False)[0]
                ref = labels_to_regions(np.load(Path(data_dir) / "native_seg" / f"{c['case_id']}.npz")["seg"])
                prob = np.mean([to_native(predict_volume(m, x.to(dev)), c) for m in models], 0)
                rows.append(case_scores(name, split, c, prob, ref))
                if i % 25 == 0:
                    print(f"{name} {split} {i}/{len(cases)}", flush=True)
            pd.DataFrame(rows).to_csv(out / "postprocess_per_case.csv", index=False)
    df = pd.DataFrame(rows)
    summary = summarise(df)
    (out / "postprocess_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
