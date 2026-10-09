# 09: 4-Stage Progressive Backbone Unfreezing, Layer-Wise Learning Rate Schedule & Soft Weak Supervision

**What to build:** Refine the DINOv2-Small SlotHead DDP training pipeline across Kaggle Dual Tesla T4 GPUs with a 4-stage progressive backbone unfreezing schedule across a 10-epoch budget, fine-grained 4-tier Layer-Wise Learning Rate Decay (LLRD), integration of continuous soft weak supervision labels (`synovitis_soft`), Stage 4-restricted Top-3 checkpoint averaging, and true validation loss telemetry.

**Blocked by:** 07: Efficiency Track Training Stabilization, Anti-Overfitting & Caching Optimization

**Status:** ready-for-agent

- [x] **4-Stage Progressive Backbone Unfreezing Schedule**:
  - In `scripts/train_ddp_efficiency.py`, update `apply_unfreezing_stage(raw_model, stage)` and the outer loop for an explicit 4-stage progression across a 10-epoch budget:
    - **Stage 1 (Epochs 1–2)**: Backbone completely frozen (`requires_grad = False`). Warm up SlotHead queries, cross-attention, and classifier head at peak $\text{LR}=1\times 10^{-3}$ decaying to $5\times 10^{-4}$.
    - **Stage 2 (Epochs 3–4)**: Unfreeze `backbone.blocks[10:]` (Top-2 blocks: 10, 11) and `backbone.norm` at $\text{LR}=1\times 10^{-5}$; SlotHead head at $\text{LR}=5\times 10^{-4} \to 3\times 10^{-4}$.
    - **Stage 3 (Epochs 5–6)**: Unfreeze `backbone.blocks[8:]` (Top-4 blocks: 8, 9, 10, 11) and `backbone.norm`; blocks 8–9 at $\text{LR}=1\times 10^{-5}$, blocks 10–11 at $\text{LR}=1.5\times 10^{-5}$; SlotHead head at $\text{LR}=2\times 10^{-4}$.
    - **Stage 4 (Epochs 7–10)**: Unfreeze all 12 backbone blocks (`backbone.blocks[0:]`) and `patch_embed`/`pos_embed`/`cls_token`. Apply 4-tier LLRD (Bottom 0–3 at $5\times 10^{-6}$, Mid 4–7 at $1\times 10^{-5}$, Mid-Top 8–9 at $1.5\times 10^{-5}$, Top 10–11 at $2\times 10^{-5}$) with a 100-step ramp and cosine annealing to zero.
- [x] **Differential Optimizer & LLRD Mapping**:
  - Update `build_differential_optimizer` in `src/training/train_efficiency.py` and `lr_for()` in `scripts/train_ddp_efficiency.py` to support 4 backbone block families (`bb_top2`, `bb_top1`, `bb_mid`, `bb_bot`) alongside `head`.
  - Maintain AdamW weight decay at `0.05` for 2D weight tensors and `0.0` for 1D biases/norms.
- [x] **Continuous Soft Weak Supervision (`synovitis_soft`)**:
  - In `scripts/train_ddp_efficiency.py`, check for `synovitis_soft` in `train_df`. For the training split, replace `train_df["Synovitis"]` with `train_df["synovitis_soft"]` (using effusion-conditional probabilities 0.63 / 0.22) while preserving binary ground truths on validation.
- [x] **Stage 4-Restricted Top-3 Checkpoint Averaging**:
  - In `scripts/train_ddp_efficiency.py`, restrict Top-3 checkpoint candidates strictly to Stage 4 (`epoch >= 7`) to prevent averaging weights across mismatched frozen/unfrozen layer regimes.
  - Retain automatic fallback to `dinov2_slothead_best_single.pth` if averaged weights do not outperform the single best epoch.
- [x] **True Validation Loss Telemetry**:
  - In `evaluate_dataset_collective`, accumulate and return true collective multi-label BCE loss alongside `score_11` and `score_12`.
  - Fix `history`, `ddp_results.json`, and TensorBoard logging so `val_loss` and `tb_logger.log_scalar("val/loss", ...)` reflect true validation loss rather than repeating `macro_auc_11`.
- [ ] **Notebook & Script Synchronization**:
  - Update `scripts/make_training_notebook.py` and regenerate `notebooks/rsna_efficiency_training.ipynb` with `--epochs 10`. (Script updated; notebook regeneration scheduled post-review).
- [x] **Verification**:
  - Unit tests verify 4-stage parameter requires_grad transitions and optimizer parameter groups.
  - Smoke test verifies DDP execution across all 4 stages without NCCL synchronization stalls or gradient clipping anomalies.
