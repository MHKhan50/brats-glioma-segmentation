"""BraTS-GLI data: download from Synapse, case discovery, patient-level split and preprocessing.

The data are never redistributed: they are downloaded into the session with the user's own Synapse access, and only
derived arrays for training stay in the private session or notebook output.
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import ndimage

MODALITIES = ("t1n", "t1c", "t2w", "t2f")
CASE_RE = re.compile(r"BraTS-GLI-(\d{5})-(\d{3})$")


# ---------------------------------------------------------------- download
def download_from_synapse(dest: str, entity: str = "syn51514105", token: Optional[str] = None,
                          name_contains: str = "Training") -> Path:
    """Download and unzip the BraTS-GLI training archive found under ``entity`` (walks sub-folders)."""
    import synapseclient
    import synapseutils

    syn = synapseclient.Synapse(silent=True)
    syn.login(authToken=token)
    dest_p = Path(dest); dest_p.mkdir(parents=True, exist_ok=True)
    chosen = []
    for _, _, files in synapseutils.walk(syn, entity):
        for name, syn_id in files:
            if name_contains.lower() in name.lower() and name.lower().endswith(".zip"):
                chosen.append((name, syn_id))
    if not chosen:
        raise FileNotFoundError(f"no zip with '{name_contains}' under {entity}")
    manifest = []
    for name, syn_id in chosen:
        f = syn.get(syn_id, downloadLocation=str(dest_p), ifcollision="keep.local")
        manifest.append({"name": name, "synapse_id": syn_id, "version": f.versionNumber, "md5": f.md5,
                         "bytes": f.fileSize})
        with zipfile.ZipFile(f.path) as z:
            z.extractall(dest_p)
        Path(f.path).unlink()
    (dest_p / "download_manifest.json").write_text(json.dumps(manifest, indent=2))
    return dest_p


# ---------------------------------------------------------------- cases and split
def find_cases(root: str) -> pd.DataFrame:
    rows = []
    for d in sorted(Path(root).rglob("BraTS-GLI-*")):
        m = CASE_RE.match(d.name)
        if not d.is_dir() or not m:
            continue
        paths = {k: d / f"{d.name}-{k}.nii.gz" for k in (*MODALITIES, "seg")}
        if all(p.exists() for p in paths.values()):
            rows.append({"case_id": d.name, "patient": m.group(1), "timepoint": m.group(2),
                         **{k: str(v) for k, v in paths.items()}})
    if not rows:
        raise FileNotFoundError(f"no complete BraTS-GLI cases under {root}")
    return pd.DataFrame(rows)


def patient_split(df: pd.DataFrame, seed: int = 0, fractions=(0.7, 0.1, 0.2)) -> pd.Series:
    """Split by patient (the 5-digit id), so different timepoints of one patient never cross splits."""
    patients = np.array(sorted(df["patient"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    a = int(round(len(patients) * fractions[0])); b = int(round(len(patients) * (fractions[0] + fractions[1])))
    assign = {**{p: "train" for p in patients[:a]}, **{p: "val" for p in patients[a:b]},
              **{p: "test" for p in patients[b:]}}
    return df["patient"].map(assign)


# ---------------------------------------------------------------- preprocessing
def brain_bbox(volumes: np.ndarray, margin: int = 2):
    """Bounding box of non-zero voxels over all modalities, as (start, stop) per axis."""
    nz = np.argwhere(np.any(volumes != 0, axis=0))
    lo = np.maximum(nz.min(0) - margin, 0)
    hi = np.minimum(nz.max(0) + 1 + margin, volumes.shape[1:])
    return [(int(a), int(b)) for a, b in zip(lo, hi)]


def normalise_to_uint8(volume: np.ndarray, mask: np.ndarray, lo: float = -4.0, hi: float = 6.0) -> np.ndarray:
    """Z-score inside the brain, clip to [lo, hi] and store as uint8 (step 10/255 SD), background 0."""
    v = volume.astype(np.float32)
    vals = v[mask]
    z = (v - vals.mean()) / (vals.std() + 1e-6) if vals.size else np.zeros_like(v)
    q = np.round((np.clip(z, lo, hi) - lo) / (hi - lo) * 254 + 1)
    q[~mask] = 0
    return q.astype(np.uint8)


def uint8_to_float(q: np.ndarray, lo: float = -4.0, hi: float = 6.0) -> np.ndarray:
    out = (q.astype(np.float32) - 1) / 254 * (hi - lo) + lo
    out[q == 0] = 0.0
    return out


def preprocess_arrays(volumes: np.ndarray, seg: np.ndarray, size: int = 128):
    """Crop to the brain, resample to size^3, normalise. Returns (image uint8 [4,S,S,S], label uint8, meta)."""
    box = brain_bbox(volumes)
    sl = tuple(slice(a, b) for a, b in box)
    crop = volumes[(slice(None),) + sl]
    lab = seg[sl]
    zoom = [size / s for s in crop.shape[1:]]
    mask = np.any(crop != 0, axis=0)
    mask_r = ndimage.zoom(mask.astype(np.uint8), zoom, order=0).astype(bool)
    img = np.stack([normalise_to_uint8(ndimage.zoom(c.astype(np.float32), zoom, order=1), mask_r) for c in crop])
    lab_r = ndimage.zoom(lab.astype(np.uint8), zoom, order=0)
    meta = {"bbox": box, "shape": list(seg.shape), "size": size}
    return img, lab_r.astype(np.uint8), meta


def preprocess_case(row: Dict, out_dir: str, size: int = 128) -> Dict:
    import nibabel as nib
    vols = np.stack([np.asarray(nib.load(row[m]).dataobj, dtype=np.float32) for m in MODALITIES])
    seg_img = nib.load(row["seg"])
    seg = np.asarray(seg_img.dataobj).astype(np.uint8)
    img, lab, meta = preprocess_arrays(vols, seg, size)
    out = Path(out_dir)
    np.save(out / "images" / f"{row['case_id']}.npy", np.concatenate([img, lab[None]], axis=0))
    np.savez_compressed(out / "native_seg" / f"{row['case_id']}.npz", seg=seg)
    meta.update(case_id=row["case_id"], spacing=[float(s) for s in seg_img.header.get_zooms()[:3]])
    return meta


def preprocess_all(root: str, out_dir: str, size: int = 128, workers: int = 4, limit: Optional[int] = None,
                   seed: int = 0) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor
    df = find_cases(root)
    df["split"] = patient_split(df, seed)
    if limit:
        df = df.groupby("split", group_keys=False).head(max(2, limit // 3)).reset_index(drop=True)
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "native_seg").mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as ex:
        metas = list(ex.map(preprocess_case, df.to_dict("records"), [out_dir] * len(df), [size] * len(df)))
    meta = pd.DataFrame(metas)
    df = df.merge(meta, on="case_id")
    df[["case_id", "patient", "timepoint", "split", "bbox", "shape", "size", "spacing"]].to_json(
        out / "cases.json", orient="records", indent=1)
    return df
