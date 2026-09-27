# 05: Single-GPU Training Pipeline & Top-3 Checkpoint Averaging

**What to build:** The core single-GPU training engine (`src/training/train_efficiency.py`) integrating the 6-slot dataset, DINOv2-Small SlotHead model, `MaskedBCEWithLogitsLoss`, and `TensorBoardLogger`. It trains on Gemini Phase 2 calibrated soft labels (`labels_soft.parquet`) across a 3-Fold Scanner-Stratified GroupKFold split, applies square-root damped positive class weights, utilizes layer-wise differential learning rates (`3e-5` backbone, `1e-3` head) with `OneCycleLR`, evaluates validation Macro ROC-AUC with gold-filtered Synovitis, and saves/averages weights from the top 3 best validation epochs.

**Blocked by:** 02: Fast 6-Slot DICOM Preprocessing Pipeline, 03: DINOv2-Small SlotHead Cross-Attention Architecture (with logger protocol support from 04)

**Status:** ready-for-agent

- [ ] Training executes cleanly on a single GPU (`cuda:0`) on Fold 0 of `cv_folds_3fold.csv`.
- [ ] Loss function applies square-root damped `pos_weight` vector and masks out NaN target values (acute tear silence).
- [ ] Differential optimizer updates backbone at `3e-5` and SlotHead/classifier at `1e-3` with weight decay 0.02.
- [ ] `OneCycleLR` schedule cycles smoothly over 12 epochs with `pct_start=0.15`.
- [ ] Validation evaluation accumulates out-of-fold predictions, computes Macro ROC-AUC, filters Synovitis to Gold Studies, and logs all metrics to TensorBoard.
- [ ] Top-3 validation epoch checkpoints are saved, and their state dictionaries are averaged post-training into an ensemble weight file `dinov2_slothead_top3.pth`.
- [ ] Integration smoke test runs 2 epochs on 20 studies, verifying checkpoint saving and loss decrease without errors.
