# 10: 336px Nyquist Resolution, 140mm Physical FOV Cropping & Calibrated Soft Weak Supervision

**What to build:** Upgrade the DINOv2-Small SlotHead DDP training pipeline to 336×336 px input resolution with 140 mm physical FOV joint cropping (satisfying the Nyquist sampling threshold for 1–3 mm tears), create a dedicated offline 336px pre-caching pipeline (`scripts/preprocess_336px_cache.py`), supervise with continuous calibrated soft targets alongside a $2.0\times$ gold study loss multiplier, and execute an extended 14-epoch 4-stage progressive unfreezing schedule with Stage-4 Top-3 checkpoint averaging targeting 0.89–0.90+ Macro AUC.

**Blocked by:** 09: 4-Stage Progressive Backbone Unfreezing, Layer-Wise Learning Rate Schedule & Soft Weak Supervision

**Status:** completed

- [x] **Offline 336px Pre-Caching Pipeline (`scripts/preprocess_336px_cache.py`)**:
  - Implement 140 mm physical millimeter cropping: compute crop box $S = \text{round}(140.0 / \text{PixelSpacing})$ with raw corner-median background fill and symmetrical square padding before resizing to $336 \times 336\text{ px}$.
  - Project physical slice normal $k = \vec{p} \cdot (\vec{r} \times \vec{c})$ to guarantee anatomical ordering across all 6 slots (`SAG_FLUID_FS`, `COR_FLUID_FS`, `AX_FLUID_FS`, `SAG_FLUID_NOFS`, `COR_T1`, `SAG_T1`).
  - Cache 5-slice depth stacks ($[c-2 \dots c+2]$) per slot into compressed `.npz` files per study to eliminate online DICOM decode bottlenecks on Dual-T4 Kaggle runs.
- [x] **DINOv2-Small SlotHead Positional Embedding Adaptation for 336px**:
  - In `src/models/dinov2_slothead.py`, ensure bicubic 2D interpolation of `pos_embed` from $(37, 37)$ (native 518px) to $(24, 24)$ ($336 \times 336\text{ px}$, patch size 14) executes seamlessly upon backbone initialization, enforced with strict length assertion.
  - Implement non-reentrant gradient checkpointing (`use_reentrant=False`) across transformer blocks so frozen early blocks do not starve downstream unfrozen blocks.
  - Verify SlotHead cross-attention queries `(12, D)` attend across the 6 slot representations with batch size 16 (8 per GPU on Dual T4) under FP16 autocast.
- [x] **Calibrated Continuous Soft Supervision & $2.0\times$ Gold Loss Weighting**:
  - In `src/training/losses.py` and `scripts/train_ddp_efficiency.py`, update `BCEWithLogitsLoss` to accept continuous posterior probability targets $[0.0, 1.0]$ across all 12 findings for pseudo-label studies (including `synovitis_soft` effusion-conditional probabilities).
  - Implement per-sample loss weighting: apply a $2.0\times$ loss multiplier to the 58 Gold ground-truth studies while keeping $1.0\times$ for pseudo-label studies.
  - Maintain class-level positive weights (`pos_weight`) derived from training prevalence.
- [x] **14-Epoch 4-Stage Progressive Unfreezing Schedule**:
  - In `scripts/train_ddp_efficiency.py` and `src/training/ddp.py`, extend training to 14 epochs across 4 distinct stages with clean reduction hook reconstruction:
    - **Stage 1 (Epochs 1–2)**: Backbone frozen. SlotHead query & classifier warmup ($\text{LR}=1\times 10^{-3} \to 5\times 10^{-4}$).
    - **Stage 2 (Epochs 3–4)**: Unfreeze `backbone.blocks[10:]` (Top-2) at $\text{LR}=1\times 10^{-5}$; head at $\text{LR}=3\times 10^{-4}$.
    - **Stage 3 (Epochs 5–7)**: Unfreeze `backbone.blocks[8:]` (Top-4); blocks 8–9 at $1\times 10^{-5}$, blocks 10–11 at $1.5\times 10^{-5}$; head at $2\times 10^{-4}$.
    - **Stage 4 (Epochs 8–14)**: Unfreeze all 12 blocks with 4-tier LLRD (Bottom at $5\times 10^{-6}$, Mid at $1\times 10^{-5}$, Mid-Top at $1.5\times 10^{-5}$, Top at $2\times 10^{-5}$) and cosine annealing to zero over 7 full epochs.
- [x] **Stage-4 Top-3 Checkpoint Averaging**:
  - Automatically track top 3 validation checkpoints in Stage 4 (Epochs 8–14).
  - Linearly average the state dictionaries into bare `dinov2_slothead_top3.pth` (no `module.` prefix).
  - Evaluate both `dinov2_slothead_best_single.pth` and `dinov2_slothead_top3.pth` on validation set and log comparisons.
- [x] **Telemetry & Kaggle Notebook Synchronization**:
  - In `scripts/make_training_notebook.py`, update arguments for `--epochs 14 --img-size 336 --gold-weight 2.0 --cache-dir /kaggle/tmp/rsna_cache_336_140mm`. (Notebook file generation deferred per supervisor instruction).
  - Ensure TensorBoard logs true validation loss, 11-label Macro AUC, 12-label Macro AUC, and Gold-subset validation AUC (NaN-safe with `gold_n_labels`).
- [x] **Verification**:
  - Unit tests verify 336px patch grid interpolation $(24, 24)$, non-reentrant gradient checkpointing, continuous soft loss calculation, and $2.0\times$ gold loss weighting (`pytest tests/test_caching_and_regularization.py`: 26 passed, 2 skipped).
  - Full test suite passed (92 passed, 4 skipped across all modules).
  - Multi-stage DDP smoke script (`scripts/smoke_ddp.py`) verified gradient flow and rank parity cleanly.
