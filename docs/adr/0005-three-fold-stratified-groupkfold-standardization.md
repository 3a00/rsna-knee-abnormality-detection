# Three-Fold Stratified GroupKFold Cross-Validation Standardization

We standardized cross-validation on a 3-fold Scanner-Fingerprint StratifiedGroupKFold split, replacing 5-fold CV to ensure full pipeline completion within Kaggle's 12-hour session limit and maximize gold-label statistical support per validation fold.

## Status

accepted

## Context

Running 5-fold training of the 2.5D MIL architecture (8 epochs per fold, 24 slices/series across Sagittal/Coronal/Axial) takes ~15 hours, causing repeated job timeouts at Fold 3 on Kaggle's 12-hour execution limit. Furthermore, with only 58 total gold studies (and rare classes like MCL having only 9 positive gold cases), 5-fold splits left ~11-12 gold cases per fold (some folds having 0-1 rare positives). 3-fold splits increase gold validation coverage to ~19-20 studies per fold, significantly improving the statistical reliability of per-label AUC evaluation.

## Considered Options

- **5-Fold Cross-Validation Across Multi-Session Checkpointing**: Requires manual session restarts, intermediate artifact uploads, and complex pipeline orchestration.
- **3-Fold Stratified GroupKFold (Selected)**: Fits entirely within a single 9–10 hour Kaggle run with a 2-hour safety margin, groups studies strictly by hardware scanner fingerprint to prevent site leakage, and stratifies on `is_gold` to ensure balanced gold distribution across all 3 folds.

## Consequences

- End-to-end 3-fold training and validation completes reliably within Kaggle's 12-hour ceiling.
- Higher sample density for rare gold labels per validation fold.
- Checkpoints `fold0_best.pth`, `fold1_best.pth`, and `fold2_best.pth` constitute the complete ensemble.
