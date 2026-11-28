# Changelog

## Unreleased

- `brats-seg postprocess`: enhancing-tumour volume rule (Isensee et al., 2018), threshold chosen on validation per ensemble, test scored once, with a paired patient-level bootstrap.
- Notebook `03_brats_postprocess_gpu.ipynb` (inference only, about 4 to 5 hours on a T4).

## Unreleased

- Training saves a checkpoint after every epoch (model, optimiser, scheduler, scaler, history) and resumes from it when a session stops; finished runs are skipped. Two data-loading workers instead of four, to lower memory use. The GPU notebook restores checkpoints from its own earlier output and tests resuming in the smoke step.

## 0.1.0 — 2026-11-21
- BraTS-GLI pipeline: Synapse download with manifest, case discovery, patient-level split, brain cropping, 128³ resampling and uint8 storage.
- Models: 3D U-Net (MONAI DynUNet) and SwinUNETR; region-based training with Dice + BCE, random 96³ crops, AMP, multi-GPU.
- Evaluation in native space: Dice, HD95 and voxel ECE per region; seed ensembles; patient-level bootstrap intervals; paired model difference.
- Kaggle notebooks for preprocessing (CPU) and training (GPU) with resumable seeds; tests and CI.
