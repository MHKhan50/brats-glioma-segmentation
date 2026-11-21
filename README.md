# Brain-tumour segmentation on BraTS-GLI: 3D U-Net vs SwinUNETR

Version `0.1.0` · Research software, not a medical device.

Adult glioma segmentation on multi-parametric MRI from the **BraTS 2023 adult glioma (BraTS-GLI)** training cohort: four co-registered sequences per case (T1, T1 with contrast, T2, T2-FLAIR) and expert labels for the non-enhancing tumour core, the surrounding FLAIR hyperintensity and the enhancing tumour. Two architectures are compared under one protocol:

- **3D U-Net**: MONAI `DynUNet` in the nnU-Net style (residual blocks, instance norm, filters 32 to 320).
- **SwinUNETR**: Swin-transformer encoder with a convolutional decoder (feature size 24, gradient checkpointing).

What makes the comparison trustworthy:

- **Patient-level split** (70/10/20 of patients, seed 0): different timepoints of one patient never fall in two splits.
- **Evaluation in the original image space**: predictions are resampled back to each case's brain box and native grid before scoring, so Dice and HD95 are comparable to published BraTS numbers.
- **Three regions** with the BraTS conventions: whole tumour (WT), tumour core (TC), enhancing tumour (ET); HD95 = 373.13 when exactly one of prediction and reference is empty.
- **Three seeds per model**, seed ensembles, 95% bootstrap intervals over patients, and the paired SwinUNETR minus U-Net difference.
- **Voxel calibration** (expected calibration error inside the brain box) next to Dice.

## Data access and governance

The data are **not redistributed** here. Each user needs their own Synapse account with access to BraTS-GLI (`syn51514105`), a personal access token, and must follow the BraTS data-use terms and cite the papers listed on the challenge page. On Kaggle the token is stored as the secret `SYNAPSE_AUTH_TOKEN`; the preprocessed arrays stay in a private notebook output. See `docs/DATA.md`.

## Run on Kaggle

1. **"BraTS-GLI Download and Preprocess (CPU)"**: import `notebooks/01_brats_preprocess_cpu.ipynb`; CPU, Internet on, secret attached. About 1.5–2 hours; output about 13 GB.
2. **"BraTS-GLI 3D U-Net vs SwinUNETR (GPU)"**: import `notebooks/02_brats_train_gpu.ipynb`; GPU T4 x2; add the output of notebook 1 as input. Each session trains the seeds named in the training cell; finished seeds are restored from attached earlier outputs and skipped.

## Run locally

```bash
python -m pip install -e ".[train,dev]"
pytest -q
export SYNAPSE_AUTH_TOKEN=...            # never commit it
brats-seg download --dest raw
brats-seg preprocess --raw raw --out prep
brats-seg train --data prep --out runs --model unet --seeds 0,1,2 --epochs 60 --batch-size 4 --val-every 10
brats-seg evaluate --data prep --runs runs --out reports
brats-seg analyse --reports reports
```

## Roadmap

| Version | Scope |
|---|---|
| **0.1** | Data pipeline, patient-level split, both models, native-space evaluation (this release) |
| 0.2 | Results for three seeds per model, calibration and failure analysis by tumour size |
| 0.3 | Robustness: missing sequences and other BraTS cohorts (meningioma, metastases, paediatric) |
| 1.0 | Paper, released weights on Hugging Face and a browser viewer for MRI slices with the predicted regions |

## Licence

Code: MIT. BraTS data: their own terms; not included.
