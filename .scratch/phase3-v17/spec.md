# Phase 3 v17 Specification: Metric Repair, Loss Rebalancing, and 3-Fold CV Baseline

**Status:** ready-for-agent

## Problem Statement

The Phase 3 v16 baseline achieved a solid 3-fold mean Macro ROC-AUC of 0.8008 ± 0.0042, but was held back by three preventable engineering issues:
1. **Metric Failure:** Synovitis reported `N/A` during validation because `roc_auc_score` crashed on continuous soft targets (0.22 / 0.63), hiding the model's true performance on this label.
2. **Class Imbalance Penalty:** Rare labels suffered from extreme negative:positive class imbalance (MCL at 48.8×, Fracture at 12.6×, Lateral OA at 9.1×), causing the model to underpredict positive cases without loss reweighting.
3. **Session Timeout Risk:** 5-fold cross-validation required ~15 hours, repeatedly timing out at Fold 3 on Kaggle's 12-hour session ceiling, while also starving rare gold labels of sufficient validation samples per fold.

## Solution

A lean, high-impact Phase 3 improvement cycle (v17) that:
1. Fixes Synovitis AUC evaluation by filtering to gold validation studies with hard binary targets.
2. Activates square-root damped positive loss weighting (`pos_weight`) in `MaskedBCEWithLogitsLoss` to penalize false negatives on rare abnormalities without gradient explosion.
3. Standardizes on a leak-free 3-Fold Scanner-Fingerprint StratifiedGroupKFold split, fitting full end-to-end training and OOF evaluation within a single 9–10 hour Kaggle Tesla T4 session.

## User Stories

1. As a machine learning practitioner, I want Synovitis validation AUC to be computed on gold ground-truth rows so that I can monitor the model's true discrimination ability across all 12 target labels.
2. As a model developer, I want the binary cross-entropy loss to apply square-root scaled `pos_weight` per label so that rare pathologies (like MCL and Fracture) receive sufficient gradient signal to learn effectively.
3. As an experimentalist, I want training to complete all cross-validation folds on a single Kaggle Tesla T4 GPU within a single session (≤ 12 hours) so that I have a complete out-of-fold validation set and full ensemble checkpoints.
4. As an evaluator, I want out-of-fold gold predictions calibrated via Platt scaling so that final ensemble probabilities are well-calibrated ranking scores.

## Implementation Decisions

- **Metric Seam:** Modify `compute_per_label_auc()` in `trainer.py` to accept an optional `gold_mask` parameter. For Synovitis only, restrict AUC computation to gold validation rows; compute all other 11 labels across all non-NaN validation rows. Macro AUC is the unweighted arithmetic mean of all 12 valid per-label AUCs.
- **Loss Rebalancing:** Add `labels.pos_weights` list to `config.yaml` computed via $\sqrt{\text{neg}/\text{pos}}$ on `pseudo_labels.csv`:
  `[2.70, 6.98, 1.21, 2.34, 2.12, 3.01, 1.83, 1.60, 2.94, 2.49, 2.21, 3.55]` (MCL capped at 6.98, all other classes between 1.21 and 3.55). Pass this tensor to `MaskedBCEWithLogitsLoss` in `trainer.py`.
- **Split Standardization:** Standardize on 3-Fold Scanner-Fingerprint StratifiedGroupKFold (`cv_folds_3fold.csv`), balancing total studies (~1,469/fold) and gold studies (~19–20/fold). Update `config.yaml` `cv.n_folds: 3` and `cv.phase3_final_csv`.
- **Ensemble & Calibration:** Train Folds 0, 1, 2 on Kaggle Tesla T4 (single-GPU). Collect OOF gold logits from the best validation epochs to fit Platt scaling temperatures.

## Testing Decisions

- **Unit Testing:** Existing 46 tests in `src/datasets/tests/test_phase3.py` must continue to pass.
- **Metric Verification:** Add unit tests verifying `compute_per_label_auc` handles `gold_mask` correctly for soft-labeled columns while ignoring non-gold soft values.
- **Loss Gradient Stability:** Smoke test `MaskedBCEWithLogitsLoss` with the new `pos_weight` vector to verify loss values are positive, finite, and backprop produces clean gradients without NaNs.
- **Split Distribution Assertions:** Assert `cv_folds_3fold.csv` contains all 4,407 studies, 0 unassigned folds, identical scanner fingerprints on only one side of any fold, and ≥ 19 gold studies per fold with both 0 and 1 classes present for all 12 labels.

## Out of Scope

- **Multi-Scale Slice Sampling (Subphase 4):** Deferred to v18 to avoid pre-cached volume cache invalidation and ensure clean ablation against v16.
- **Laterality Embeddings:** Deferred to Phase 4 (requires extracting `ImageLaterality` DICOM tags).
- **Slice Mixup:** Omitted due to mathematical conflict with NaN silence masking.

## Further Notes

- Target benchmark: exceed v16 3-fold mean macro AUC of 0.8008 ± 0.0042 (targeting ≥ 0.815+).
- Reference ADRs: [ADR-0001](file:///home/wenalz/Documents/Antigravity%20projects/RSNA_MRI/docs/adr/0001-two-tier-llm-weak-label-extraction-and-imputation.md), [ADR-0002](file:///home/wenalz/Documents/Antigravity%20projects/RSNA_MRI/docs/adr/0002-2.5d-mil-with-attention-pooling.md), [ADR-0003](file:///home/wenalz/Documents/Antigravity%20projects/RSNA_MRI/docs/adr/0003-single-gpu-execution-over-dataparallel.md), [ADR-0004](file:///home/wenalz/Documents/Antigravity%20projects/RSNA_MRI/docs/adr/0004-tesla-t4-selection-over-p100.md), [ADR-0005](file:///home/wenalz/Documents/Antigravity%20projects/RSNA_MRI/docs/adr/0005-three-fold-stratified-groupkfold-standardization.md).
