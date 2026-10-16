# 11: Multi-Fold Pipeline Runner (Folds 1 & 2), Percentile Rank OOF Evaluation & Single-Model Refinement

**What to build:** Extend the DINOv2-Small SlotHead DDP training pipeline to support multi-fold execution (`--folds 1,2`), train Folds 1 and 2 sequentially in a single Kaggle Dual-T4 session reusing the existing 336px volume cache, integrate the retained Fold 0 weights (`0.8465` AUC), and evaluate the full 3-fold Out-Of-Fold (OOF) ensemble using both standard probability averaging and Percentile Rank Averaging across all 4,407 studies.

**Blocked by:** 10: 336px Nyquist Resolution, 140mm Physical FOV Cropping & Calibrated Soft Weak Supervision

**Status:** open

- [ ] **Multi-Fold Training Support (`scripts/train_ddp_efficiency.py`)**:
  - Extend CLI parser to support `--folds` (comma-separated string e.g. `'1,2'` or `'0,1,2'`) alongside single `--fold`.
  - In `train_ddp_worker`, loop sequentially through the requested folds within the same DDP session context:
    - Re-initialize model weights from `dinov2_small_backbone.pth` for each fold to prevent inter-fold weight leakage.
    - Reset optimizer, LR schedules, and early stage boundaries per fold.
    - Save fold-specific checkpoints: `dinov2_slothead_fold{k}_best.pth` and `dinov2_slothead_fold{k}_top3.pth`.
    - Record comprehensive multi-fold history in `outputs/weights/multifold_results.json`.
- [ ] **Full 3-Fold Out-of-Fold (OOF) Evaluator (`src/training/metrics.py` & `scripts/evaluate_3fold_oof.py`)**:
  - Build `scripts/evaluate_3fold_oof.py` to ingest validation prediction records from all 3 folds:
    - Retained Fold 0 checkpoint: `outputs/weights/dinov2_slothead_fold0_best.pth` (from run v6).
    - Newly trained Fold 1 checkpoint: `dinov2_slothead_fold1_best.pth`.
    - Newly trained Fold 2 checkpoint: `dinov2_slothead_fold2_best.pth`.
  - Assemble complete $N=4,407$ OOF prediction matrix:
    - Evaluate raw sigmoid probability Macro AUC (11-label, 12-label, Gold58).
    - Evaluate **Percentile Rank Averaged** Macro AUC via `rank(method='average', pct=True)` per fold before ensembling.
    - Quantify the exact ensemble delta (+0.020 to +0.030 target).
- [ ] **Telemetry & Kaggle Notebook Synchronization (`scripts/make_training_notebook.py`)**:
  - Update `scripts/make_training_notebook.py` to:
    - Include retained Fold 0 weights in the codebase package or download them.
    - Set Cell 5 command to execute `--folds 1,2`.
    - Update Cell 6 to execute the full 3-fold OOF evaluation and copy all 3 fold weights (`dinov2_slothead_fold{0,1,2}_best.pth`) to `/kaggle/working/` for Kaggle dataset publishing.
- [ ] **Verification**:
  - Unit tests verify multi-fold CLI argument parsing, clean weight re-initialization between folds, and percentile rank averaging calculation.
  - Multi-fold dry run verifies sequential execution without CUDA memory leaks across fold boundaries.
