# 11: Multi-Fold Pipeline Runner (Folds 1 & 2), Percentile Rank OOF Evaluation & Single-Model Refinement

**What to build:** Extend the DINOv2-Small SlotHead DDP training pipeline to support multi-fold execution (`--folds 1,2`), train Folds 1 and 2 sequentially in a single Kaggle Dual-T4 session reusing the existing 336px volume cache, integrate the retained Fold 0 weights (`0.8465` AUC), and evaluate the full 3-fold Out-Of-Fold (OOF) ensemble using both standard probability averaging and Percentile Rank Averaging across all 4,407 studies.

**Blocked by:** 10: 336px Nyquist Resolution, 140mm Physical FOV Cropping & Calibrated Soft Weak Supervision

**Status:** completed

- [x] **Multi-Fold Training Support (`scripts/train_ddp_efficiency.py`)**:
  - Extend CLI parser to support `--folds` (comma-separated string e.g. `'1,2'` or `'0,1,2'`) alongside single `--fold`.
  - In `train_ddp_worker`, loop sequentially through the requested folds within the same DDP session context:
    - Re-initialize model weights from `dinov2_small_backbone.pth` for each fold to prevent inter-fold weight leakage.
    - Reset optimizer, LR schedules, and early stage boundaries per fold.
    - Save fold-specific checkpoints: `dinov2_slothead_fold{k}_best.pth` and `dinov2_slothead_fold{k}_top3.pth`.
    - Record comprehensive multi-fold history in `outputs/weights/multifold_results.json`.
- [x] **Full 3-Fold Out-of-Fold (OOF) Evaluator (`src/training/metrics.py` & `scripts/evaluate_3fold_oof.py`)**:
  - Build `scripts/evaluate_3fold_oof.py` to ingest validation prediction records from all 3 folds (Folds 0, 1, 2) trained on the authoritative 14-epoch 4-stage recipe.
  - Assemble complete Out-of-Fold (OOF) prediction matrix across all verified cached studies:
    - Per-fold Macro AUC (mean ± std, ddof=1) is the headline cross-validation benchmark statistic.
    - Evaluate pooled raw sigmoid probability Macro AUC (11-label, 12-label, Gold 58).
    - Evaluate per-fold Percentile Rank Normalized Macro AUC via `(rankdata(X, method='average', axis=0) - 1) / (N - 1)` to measure cross-fold calibration alignment delta (no ensembling in OOF).
- [x] **Full Model Visualization & Telemetry (`src/utils/tensorboard.py` & `scripts/train_ddp_efficiency.py`)**:
  - On Rank 0 at model initialization, trace and log the complete computational graph (`tb_logger.log_graph(model, dummy_batch)`) to visualize multi-view routing, DINOv2 backbone, SlotHead attention pooling, and classification heads (best-effort).
  - Log weight and gradient distributions (`writer.add_histogram`) at stage boundaries (`epoch in (2, 4, 7, epochs)`) with `bins=64` to monitor parameter shifts while keeping event files compact.
  - Log fold-prefixed PR curves per fold (`fold{k}/PR/{finding}`). Attention heatmaps are out of scope (deferred to post-hoc analysis).
- [x] **Telemetry & Kaggle Notebook Synchronization (`scripts/make_training_notebook.py`)**:
  - Update `scripts/make_training_notebook.py` to:
    - Add Cell 3b: fail-fast preflight validating splits, labels, scripts, backbone path, and `/kaggle/tmp` disk space before caching.
    - Set Cell 5 command to execute `--folds 0,1,2` under unbuffered live streaming (`run_live`).
    - Mount legacy v6 Fold 0 weights only as a reference (`dinov2_slothead_fold0_v6_reference.pth`).
    - Update Cell 6 to execute the full 3-fold OOF evaluation and verify all 3 fold weights (`dinov2_slothead_fold{0,1,2}_best.pth`) in `outputs/weights/`.
- [x] **Verification**:
  - Unit tests verify multi-fold CLI argument parsing, clean weight re-initialization from base timm backbone between folds, model graph tracing on real `DINOv2SlotHead`, stage boundary consistency, and percentile rank averaging calculation.
  - Multi-fold dry run verifies sequential execution with atomic checkpoint resumption without CUDA memory leaks across fold boundaries.
