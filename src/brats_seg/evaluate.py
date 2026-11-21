"""Test-set evaluation in the original image space: Dice and HD95 per region, voxel calibration, seed ensembles."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import dice, ece, hd95
from .regions import REGIONS, enforce_nesting, labels_to_regions


def to_native(prob, meta) -> np.ndarray:
    """Resample (3, S, S, S) probabilities back to the brain box and paste them into the original grid."""
    import torch
    import torch.nn.functional as F
    box = meta["bbox"]
    size = [b - a for a, b in box]
    p = F.interpolate(prob[None].float(), size=size, mode="trilinear", align_corners=False)[0].cpu().numpy()
    full = np.zeros((3, *meta["shape"]), np.float32)
    full[(slice(None),) + tuple(slice(a, b) for a, b in box)] = p
    return full


def evaluate_models(data_dir: str, run_dir: str, out_dir: str, names=("unet", "swinunetr"), seeds=(0, 1, 2),
                    limit: int = 0) -> pd.DataFrame:
    import torch

    from .train import CaseDataset, build_model, predict_volume
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = [c for c in json.loads((Path(data_dir) / "cases.json").read_text()) if c["split"] == "test"]
    if limit:
        cases = cases[:limit]
    run, out = Path(run_dir), Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        models = []
        for s in seeds:
            ck = run / f"{name}_s{s}_best.pt"
            if ck.exists():
                m = build_model(name)
                m.load_state_dict(torch.load(ck, map_location="cpu")["state_dict"])
                models.append((s, m.to(dev).eval()))
        if not models:
            continue
        for c in cases:
            x, _ = CaseDataset([str(Path(data_dir) / "images" / f"{c['case_id']}.npy")], False)[0]
            ref = labels_to_regions(np.load(Path(data_dir) / "native_seg" / f"{c['case_id']}.npz")["seg"])
            probs = []
            for s, m in models:
                p = to_native(predict_volume(m, x.to(dev)), c)
                probs.append(p)
                rows.append(_score(name, str(s), c, p, ref))
            if len(probs) > 1:
                rows.append(_score(name, "ensemble", c, np.mean(probs, 0), ref))
        pd.DataFrame(rows).to_csv(out / "test_per_case.csv", index=False)
    return pd.DataFrame(rows)


def _score(name, seed, case, prob, ref) -> dict:
    pred = enforce_nesting(prob >= 0.5)
    sp = tuple(case.get("spacing", (1.0, 1.0, 1.0)))
    row = {"model": name, "seed": seed, "case_id": case["case_id"], "patient": case["patient"]}
    box = tuple(slice(a, b) for a, b in case["bbox"])
    for r, reg in enumerate(REGIONS):
        row[f"dice_{reg}"] = dice(pred[r], ref[r])
        row[f"hd95_{reg}"] = hd95(pred[r], ref[r], sp)
        row[f"ece_{reg}"] = ece(prob[r][box], ref[r][box])
        row[f"vol_ref_{reg}"] = int(ref[r].sum())
    return row


def summarise(per_case: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean over seeds (mean and SD of per-seed means), ensemble with patient-level bootstrap CIs, and the
    paired SwinUNETR - U-Net difference of ensembles."""
    rng = np.random.default_rng(seed)
    cols = [c for c in per_case.columns if c.startswith(("dice_", "hd95_", "ece_"))]
    out = {}
    for name, g in per_case.groupby("model"):
        seeds = g[g["seed"] != "ensemble"].groupby("seed")[cols].mean()
        out[name] = {"per_seed_mean": seeds.mean().round(4).to_dict(), "per_seed_sd": seeds.std(ddof=1).round(4).to_dict()
                     if len(seeds) > 1 else {}, "seeds": len(seeds)}
        ens = g[g["seed"] == "ensemble"] if (g["seed"] == "ensemble").any() else g
        pats = ens["patient"].unique()
        boots = {c: [] for c in cols}
        for _ in range(n_boot):
            pick = rng.choice(pats, len(pats))
            sample = pd.concat([ens[ens["patient"] == p] for p in pick])
            for c in cols:
                boots[c].append(sample[c].mean())
        out[name]["ensemble"] = {c: [round(float(ens[c].mean()), 4), round(float(np.percentile(boots[c], 2.5)), 4),
                                     round(float(np.percentile(boots[c], 97.5)), 4)] for c in cols}
    names = sorted(out)
    if {"unet", "swinunetr"} <= set(names):
        a = per_case[(per_case["model"] == "swinunetr") & (per_case["seed"] == "ensemble")].set_index("case_id")
        b = per_case[(per_case["model"] == "unet") & (per_case["seed"] == "ensemble")].set_index("case_id")
        if len(a) and len(b):
            common = a.index.intersection(b.index)
            diff = (a.loc[common, [f"dice_{r}" for r in REGIONS]] - b.loc[common, [f"dice_{r}" for r in REGIONS]])
            pat = a.loc[common, "patient"]
            pats = pat.unique(); boots = []
            for _ in range(n_boot):
                pick = rng.choice(pats, len(pats))
                boots.append(pd.concat([diff[pat == p] for p in pick]).mean().to_numpy())
            boots = np.array(boots)
            out["swinunetr_minus_unet_dice"] = {r: [round(float(diff[f"dice_{r}"].mean()), 4),
                                                    round(float(np.percentile(boots[:, i], 2.5)), 4),
                                                    round(float(np.percentile(boots[:, i], 97.5)), 4)]
                                                for i, r in enumerate(REGIONS)}
    return out


def summary_markdown(summary: dict) -> str:
    lines = ["| Model | Dice WT | Dice TC | Dice ET | HD95 WT | HD95 TC | HD95 ET | seeds |", "|---|---|---|---|---|---|---|---|"]
    for name, v in summary.items():
        if name.endswith("_dice"):
            continue
        m, sd = v["per_seed_mean"], v["per_seed_sd"]
        cell = lambda k: f"{m[k]:.3f}" + (f" ± {sd[k]:.3f}" if sd else "")
        lines.append(f"| {name} | " + " | ".join(cell(f"dice_{r}") for r in REGIONS) + " | "
                     + " | ".join(f"{m[f'hd95_{r}']:.1f}" for r in REGIONS) + f" | {v['seeds']} |")
    if "swinunetr_minus_unet_dice" in summary:
        d = summary["swinunetr_minus_unet_dice"]
        lines += ["", "SwinUNETR minus U-Net (ensembles, Dice): " + ", ".join(
            f"{r} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for r, v in d.items())]
    return "\n".join(lines) + "\n"
