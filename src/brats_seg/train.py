"""Training: 3D U-Net (MONAI DynUNet, nnU-Net style) and SwinUNETR on the three nested BraTS regions."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from .data import uint8_to_float
from .regions import labels_to_regions

MODELS = {
    "unet": {"lr": 1e-3},
    "swinunetr": {"lr": 2e-4},
}


def build_model(name: str):
    from monai.networks.nets import DynUNet, SwinUNETR
    if name == "unet":
        kernels = [[3, 3, 3]] * 6
        strides = [[1, 1, 1]] + [[2, 2, 2]] * 5
        return DynUNet(spatial_dims=3, in_channels=4, out_channels=3, kernel_size=kernels, strides=strides,
                       upsample_kernel_size=strides[1:], filters=[32, 64, 128, 256, 320, 320],
                       norm_name="instance", deep_supervision=False, res_block=True)
    if name == "swinunetr":
        try:
            return SwinUNETR(in_channels=4, out_channels=3, feature_size=24, use_checkpoint=True, spatial_dims=3)
        except TypeError:  # older MONAI needs img_size
            return SwinUNETR(img_size=(96, 96, 96), in_channels=4, out_channels=3, feature_size=24,
                             use_checkpoint=True, spatial_dims=3)
    raise ValueError(name)


class CaseDataset:
    """Reads the preprocessed uint8 arrays; random 96^3 crops, flips and intensity jitter for training."""

    def __init__(self, files, train: bool, crop: int = 96, seed: int = 0):
        self.files, self.train, self.crop = list(files), train, crop
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import torch
        arr = np.load(self.files[i], mmap_mode="r")
        img = uint8_to_float(np.asarray(arr[:4]))
        reg = labels_to_regions(np.asarray(arr[4])).astype(np.float32)
        if self.train:
            rng = np.random.default_rng(self.rng.integers(1 << 31) + i)
            c, s = self.crop, img.shape[1]
            fg = np.argwhere(reg[0] > 0)
            if len(fg) and rng.random() < 0.67:  # two thirds of crops centred near tumour
                centre = fg[rng.integers(len(fg))]
                start = np.clip(centre - c // 2, 0, s - c)
            else:
                start = rng.integers(0, s - c + 1, size=3)
            sl = tuple(slice(int(a), int(a) + c) for a in start)
            img, reg = img[(slice(None),) + sl], reg[(slice(None),) + sl]
            for ax in (1, 2, 3):
                if rng.random() < 0.5:
                    img, reg = np.flip(img, ax), np.flip(reg, ax)
            img = img * rng.uniform(0.9, 1.1, (4, 1, 1, 1)) + rng.uniform(-0.1, 0.1, (4, 1, 1, 1))
        return torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)), \
            torch.from_numpy(np.ascontiguousarray(reg, dtype=np.float32))


def predict_volume(model, img, roi: int = 96, overlap: float = 0.5):
    """Sigmoid probabilities (3, S, S, S) for one preprocessed image, sliding window with Gaussian blending."""
    import torch
    from monai.inferers import sliding_window_inference
    with torch.no_grad(), torch.autocast("cuda", enabled=torch.cuda.is_available()):
        logits = sliding_window_inference(img[None], (roi, roi, roi), 4, model, overlap=overlap, mode="gaussian")
    return torch.sigmoid(logits.float())[0]


def train_one(name: str, data_dir: str, out_dir: str, seed: int = 0, epochs: int = 100, batch_size: int = 2,
              val_every: int = 5, limit_steps: int = 0, max_val_cases: int = 0) -> dict:
    import torch
    from monai.losses import DiceLoss
    from monai.utils import set_determinism

    from .metrics import dice
    set_determinism(seed)
    torch.manual_seed(seed)
    cases = json.loads((Path(data_dir) / "cases.json").read_text())
    files = {s: [str(Path(data_dir) / "images" / f"{c['case_id']}.npy") for c in cases if c["split"] == s]
             for s in ("train", "val")}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(name).to(dev)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    loader = torch.utils.data.DataLoader(CaseDataset(files["train"], True, seed=seed), batch_size=batch_size,
                                         shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_set = CaseDataset(files["val"][:max_val_cases] if max_val_cases else files["val"], False)
    dice_loss = DiceLoss(sigmoid=True, smooth_nr=1e-5, smooth_dr=1e-5)
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=MODELS[name]["lr"], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = f"{name}_s{seed}"
    meta_path, last_path = out / f"{tag}_meta.json", out / f"{tag}_last.pt"
    if meta_path.exists():
        done = json.loads(meta_path.read_text())
        if done.get("finished") and done.get("epochs") == epochs:
            print(f"{tag}: already trained ({epochs} epochs), skipping", flush=True)
            return {k: v for k, v in done.items() if k != "history"}
    best, history, t0, start, spent = -1.0, [], time.time(), 1, 0.0
    if last_path.exists():  # resume after an interrupted session
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        (model.module if hasattr(model, "module") else model).load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        best, history, start, spent = ck["best"], ck["history"], ck["epoch"] + 1, ck.get("minutes", 0.0)
        print(f"{tag}: resuming at epoch {start} (best val mean Dice so far {best:.4f})", flush=True)
    for epoch in range(start, epochs + 1):
        model.train(); losses = []
        for step, (x, y) in enumerate(loader):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                logits = model(x)
            loss = dice_loss(logits.float(), y) + bce(logits.float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            losses.append(float(loss))
            if limit_steps and step + 1 >= limit_steps:
                break
        sched.step()
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), "minutes": round(spent + (time.time() - t0) / 60, 1)}
        if epoch % val_every == 0 or epoch == epochs:
            model.eval(); scores = []
            for i in range(len(val_set)):
                x, y = val_set[i]
                p = predict_volume(model, x.to(dev)).cpu().numpy() >= 0.5
                scores.append([dice(p[r], y[r].numpy()) for r in range(3)])
            rec["val_dice"] = [float(v) for v in np.mean(scores, 0)]
            rec["val_mean_dice"] = float(np.mean(scores))
            if rec["val_mean_dice"] > best:
                best = rec["val_mean_dice"]
                state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save({"state_dict": state, "model": name, "seed": seed, "epoch": epoch}, out / f"{tag}_best.pt")
        history.append(rec)
        print(json.dumps(rec), flush=True)
        state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save({"state_dict": state, "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "history": history,
                    "minutes": rec["minutes"]}, str(last_path) + ".tmp")
        os.replace(str(last_path) + ".tmp", last_path)  # a kill during saving cannot corrupt the checkpoint
    meta = {"model": name, "seed": seed, "epochs": epochs, "batch_size": batch_size, "best_val_mean_dice": best,
            "params": int(sum(p.numel() for p in model.parameters())),
            "minutes": round(spent + (time.time() - t0) / 60, 1), "finished": True, "history": history}
    meta_path.write_text(json.dumps(meta, indent=2))
    return {k: v for k, v in meta.items() if k != "history"}
