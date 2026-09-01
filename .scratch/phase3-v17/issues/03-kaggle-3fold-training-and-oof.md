# 03: Kaggle 3-Fold Training Run, Codebase Packaging, and Out-of-Fold Analysis

**Status:** ready-for-user-execution

## 1. Summary of Built & Uploaded Assets

1. **Codebase Packaging & Zip Files:**
   - Local archives generated in [`outputs/codebase_zip/`](file:///home/wenalz/Documents/Antigravity%20projects/RSNA_MRI/outputs/codebase_zip/):
     - `src.zip` (63 KB) — all source modules with `trainer.py` metric fix and `pos_weights` support
     - `scripts.zip` (18 KB) — data preprocessing and fingerprint scripts
     - `data_splits.zip` (353 KB) — 3-fold and 5-fold CV splits
     - `rsna_mri_codebase_v17.zip` (2.2 MB) — complete clean project snapshot
   - Pushed new version to Kaggle Dataset: [`i3abd1/rsna-mri-codebase`](https://www.kaggle.com/datasets/i3abd1/rsna-mri-codebase) containing `config.yaml`, `src/`, `scripts/`, `data/splits/cv_folds_3fold.csv`, `data/labels/`, and `data/raw/`.

2. **Phase 3 v17 Training Notebook:**
   - Created locally at [`notebooks/rsna_phase3_v17.ipynb`](file:///home/wenalz/Documents/Antigravity%20projects/RSNA_MRI/notebooks/rsna_phase3_v17.ipynb) and [`notebooks/eda/rsna_phase3_v17.ipynb`](file:///home/wenalz/Documents/Antigravity%20projects/RSNA_MRI/notebooks/eda/rsna_phase3_v17.ipynb).
   - Uploaded to Kaggle Kernel: [`i3abd1/rsna-phase-3-v17-3fold-gpu-t4`](https://www.kaggle.com/code/i3abd1/rsna-phase-3-v17-3fold-gpu-t4).
   - Configured with:
     - **Cell 1:** Environment setup, codebase extraction, pip install `timm`, `albumentations`, `pydicom`.
     - **Cell 2:** DICOM fingerprint scan (fast-loads pre-scanned `dicom_fingerprints.csv`).
     - **Cell 3:** 3-Fold Scanner-Fingerprint CV Split loading (`cv_folds_3fold.csv`, ~1,469 studies & ~19–20 gold per fold).
     - **Cell 4:** One-time DICOM preprocessing cache to `/kaggle/tmp/rsna_cache` (fast uint8 `.npz`).
     - **Cell 5:** 3-Fold 2.5D MIL Training with square-root `pos_weight` loss rebalancing and gold-restricted Synovitis AUC logging.
     - **Cell 6:** Platt temperature scaling on OOF gold logits, exporting `platt_temperature.pth` and `phase3_cv_results.json`.

---

## 2. User Manual Execution Guide (Kaggle)

> [!IMPORTANT]
> The notebook has been pushed to Kaggle but **not executed**. Follow these steps to run training on Kaggle:

1. Open notebook editor: [i3abd1/rsna-phase-3-v17-3fold-gpu-t4](https://www.kaggle.com/code/i3abd1/rsna-phase-3-v17-3fold-gpu-t4).
2. **Session Settings:**
   - **Accelerator:** GPU T4 (single GPU)
   - **Internet:** ON (for `timm`, `albumentations`, `pydicom`)
3. **Verify Attached Inputs:**
   - `rsna-knee-abnormality-detection` (Competition data)
   - `i3abd1/rsna-mri-codebase` (Codebase dataset)
4. Click **"Run All"** or execute cells 1 through 6 sequentially:
   - Cell 4 (Preprocessing cache): ~30–45 minutes (run once per session).
   - Cell 5 (3-Fold Training): ~2.5–3 hours total (~50 min per fold on Tesla T4).
   - Cell 6 (OOF & Platt Calibration): ~10 seconds.

---

## 3. Post-Training Validation & Verification Checklist

- [x] Codebase packaged into `.zip` files and pushed to Kaggle Dataset `i3abd1/rsna-mri-codebase`.
- [x] Training notebook created (`notebooks/rsna_phase3_v17.ipynb`) and pushed to `i3abd1/rsna-phase-3-v17-3fold-gpu-t4`.
- [ ] User runs notebook on Kaggle Tesla T4 (Internet ON).
- [ ] Verify all 3 folds finish successfully without session timeout.
- [ ] Verify Synovitis AUC outputs a real numeric score (expected ~0.75–0.85 on gold subset).
- [ ] Confirm 3-Fold Mean Macro ROC-AUC $\ge 0.815$ (vs v16 baseline 0.8008 ± 0.0042) and MCL AUC $\ge 0.70$.
- [ ] Confirm `platt_temperature.pth` and checkpoint files (`fold0_best.pth`, `fold1_best.pth`, `fold2_best.pth`) are saved in outputs.
