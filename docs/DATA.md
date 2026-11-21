# Data record: BraTS 2023 adult glioma (BraTS-GLI)

| Field | Record |
|---|---|
| Source | Synapse, BraTS-GLI data folder `syn51514105` (access granted per user) |
| Content | Training cohort of adult glioma cases; per case four MRI sequences (`t1n`, `t1c`, `t2w`, `t2f`) and a label map |
| Labels | 0 background, 1 non-enhancing tumour core, 2 surrounding FLAIR hyperintensity, 3 enhancing tumour |
| Regions scored | WT = {1, 2, 3}, TC = {1, 3}, ET = {3} |
| Case ids | `BraTS-GLI-PPPPP-TTT`: patient `PPPPP`, timepoint `TTT`; the split is by patient |
| Provenance | The download writes `download_manifest.json` with the Synapse id, version and MD5 of each archive |
| Redistribution | None. Raw data stay in the session; preprocessed arrays only in a private notebook output |
| Citation | Cite the papers listed on the BraTS challenge page on Synapse, as required by the data terms |

Preprocessing (`brats_seg.data.preprocess_arrays`): crop to the non-zero bounding box of all sequences (2-voxel margin), resample to 128³ (linear for images, nearest for labels), z-score each sequence inside the brain, clip to [-4, 6] SD and store as uint8 (step 0.04 SD). The original labels, the bounding box and the voxel spacing are kept so that predictions are scored in the original grid.
