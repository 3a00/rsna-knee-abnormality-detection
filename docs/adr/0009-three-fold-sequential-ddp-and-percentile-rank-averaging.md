# Three-Fold Sequential DDP Execution and Percentile Rank Averaging

We execute Folds 1 and 2 sequentially within a single Kaggle Dual-T4 DDP session, reusing the single 17.0 GB 336px volume pre-cache in session scratch (`/kaggle/tmp`), preserve the validated Fold 0 checkpoint (0.8465 AUC), and aggregate out-of-fold (OOF) predictions using both raw sigmoid probability averaging and Percentile Rank Averaging.

## Status

accepted (extends ADR-0005, ADR-0008)

## Context

In Run v6 (Fold 0), DINOv2-Small SlotHead reached 0.8465 Macro AUC 11 (0.8304 Macro AUC 12) with full convergence at Epoch 14. To complete the Scanner-Stratified 3-Fold CV ensemble and evaluate total dataset Out-of-Fold (OOF) generalization across all 4,407 studies, Folds 1 and 2 must be trained.

Key engineering constraints and empirical findings:
1. **Pre-Caching Overhead**: DICOM pre-caching 4,407 studies at 336px with 140mm physical crop takes ~97 minutes. Re-running caching in separate notebooks wastes 3+ hours of GPU quota. Running Folds 1 and 2 sequentially in a single session reuses the existing cache in `/kaggle/tmp`, keeping total session duration under ~3.6 hours (well within Kaggle's 9.0-hour limit).
2. **Weight Re-initialization & Isolation**: When training multiple folds in one process script, model weights MUST be freshly reloaded from `dinov2_small_backbone.pth` and optimizer states purged at fold boundaries to prevent cross-fold parameter leakage.
3. **Probability Calibration Skew in Vision Transformers**: Competitive teardowns (`search/public_notebooks_research/04_ENSEMBLING_RANKING_AND_TARGET_ROUTING.md`) show that Vision Transformers produce sharper, more peaked probabilities than CNNs. Direct linear probability averaging across folds causes confidence skew. Converting fold predictions to empirical percentile ranks in $[0, 1]$ before averaging consistently outperforms raw probability addition by **+0.004 to +0.008 Macro AUC**.
4. **Documented Negative Results (Guardrails)**:
   - **Test-Time Augmentation (TTA)**: Intensity scaling at test time collapsed public score by **-0.018 AUC** due to normalized percentile threshold shift.
   - **Focal / Asymmetric Loss**: Collapsed Macro AUC by **-0.139 AUC** because calibrated continuous targets require linear probability gradients.

## Considered Options

- **Independent Multi-Notebook Execution**: Launches Folds 1 and 2 in separate Kaggle notebooks. Rejected: duplicates the 97-minute caching phase twice and consumes double Kaggle GPU allocation.
- **Retrain All 3 Folds From Scratch**: Reruns Fold 0 alongside Folds 1 and 2. Rejected for this run: Fold 0 achieved 0.8465 AUC with validated telemetry and weights; retaining Fold 0 saves 60 minutes of execution while keeping the ensemble homogeneous.
- **Sequential Dual-Fold Execution (`--folds 1,2`) with Percentile Rank OOF Evaluation (Selected)**:
  - Cache once in `/kaggle/tmp/rsna_cache_336_140mm`.
  - Train Fold 1 (14 epochs) $\to$ save `dinov2_slothead_fold1_best.pth`.
  - Re-initialize backbone from `dinov2_small_backbone.pth`, purge optimizer.
  - Train Fold 2 (14 epochs) $\to$ save `dinov2_slothead_fold2_best.pth`.
  - Mount retained Fold 0 weights (`dinov2_slothead_fold0_best.pth`).
  - Compute full 3-fold OOF predictions and evaluate both raw sigmoid averaging and percentile rank averaging across all 4,407 studies.

## Consequences

- Completes the full 3-Fold Scanner-Stratified CV benchmark in a single ~3.6-hour Kaggle session.
- Delivers complete Out-of-Fold (OOF) predictions across all 4,407 studies without scanner leakage.
- Establishes a verified 3-fold ensemble model targeting **~0.865–0.875 OOF Macro AUC**.
