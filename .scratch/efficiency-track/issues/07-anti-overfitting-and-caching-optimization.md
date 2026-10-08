# 07: Efficiency Track Training Stabilization, Anti-Overfitting & Caching Optimization

**What to build:** Comprehensive training optimization, anti-overfitting regularization, and I/O caching suite for the DINOv2-Small SlotHead architecture across Kaggle dual Tesla T4 GPUs. Resolves the 8.35-hour I/O bottleneck by pre-caching 5-slice volumes in session scratch (`/kaggle/tmp/rsna_cache`), breaks input memorization via dynamic 2.5D depth jitter and symmetry-preserving MSK augmentations, implements a 3-stage progressive backbone unfreezing schedule, adds cross-attention and pre-classifier dropout, and updates the Kaggle training notebook (`notebooks/rsna_efficiency_training.ipynb`) to train in $\le 35$ minutes without overfitting.

**Blocked by:** 02: Fast 6-Slot DICOM Preprocessing Pipeline, 03: DINOv2-Small SlotHead Cross-Attention Architecture, 06: Dual-T4 DistributedDataParallel (DDP) Engine

**Status:** ready-for-agent

- [ ] **5-Slice Pre-Caching Engine**:
  - Update `src/datasets/efficiency_pipeline.py` and `scripts/preprocess_volumes.py` to extract 5 consecutive physical slices around the center per present slot (shape `(6, 5, 336, 336)` float16).
  - Persist precomputed studies as dictionary `{ "image": tensor, "presence_mask": mask, "meta": {"target_size": 336, "crop_mm": 140.0, "n_slices_cached": 5} }` in `/kaggle/tmp/rsna_cache/{uid}.pt`.
  - Validate cache provenance in `EfficiencyStudyDataset` to reject stale size/FOV artifacts.
- [ ] **Dynamic 2.5D Depth Jitter**:
  - In `EfficiencyStudyDataset` (`src/training/train_efficiency.py`), sample a random 3-slice sub-window during training (`[0:3]`, `[1:4]`, or `[2:5]`) from the cached 5-slice tensor.
  - In validation mode, deterministically select the center 3 slices (`[1:4]`).
- [ ] **Symmetry-Preserving MSK Augmentations**:
  - Implement a CPU DataLoader transformation pipeline applying mild rotation ($\pm 7^\circ$), subtle scale ($0.93\text{--}1.07$), translation ($\pm 4\%$), Gaussian blur/noise ($p=0.2$), and random brightness/contrast ($\pm 10\%$).
  - Strictly prohibit horizontal and vertical flips to avoid corrupting medial vs. lateral compartmental ground truths.
  - Apply identical spatial transformations across all 3 slices of each sequence slot.
- [ ] **Model & Head Regularization**:
  - In `src/models/dinov2_slothead.py`, wire `dropout = 0.2` into `nn.MultiheadAttention`.
  - Add `nn.Dropout(0.3)` immediately before the linear classification projection `(attended * cls_w)`.
  - In `build_differential_optimizer`, increase AdamW weight decay to `0.05` for 2D weight tensors.
- [ ] **3-Stage Progressive Backbone Unfreezing**:
  - In `scripts/train_ddp_efficiency.py`, implement a 3-stage unfreezing schedule across an 8-epoch budget:
    - **Stage 1 (Epochs 1–2)**: Backbone completely frozen (`requires_grad = False`). Warm up SlotHead queries and heads at peak $\text{LR}=1\times 10^{-3}$.
    - **Stage 2 (Epochs 3–4)**: Unfreeze `backbone.blocks[10:]` and `backbone.norm` at low $\text{LR}=1\times 10^{-5}$, SlotHead at $\text{LR}=5\times 10^{-4}$.
    - **Stage 3 (Epochs 5–8)**: Unfreeze all 12 backbone blocks with differential LR (lower on bottom blocks, peak $2\times 10^{-5}$ on top blocks) decaying to 0 via OneCycleLR/Cosine.
- [ ] **Kaggle Notebook Synchronization (`notebooks/rsna_efficiency_training.ipynb`)**:
  - Add a dedicated preprocessing cell executing `scripts/preprocess_volumes.py` to populate `/kaggle/tmp/rsna_cache` (~15–18 min one-time run).
  - Update DDP execution command to pass `--cache-dir /kaggle/tmp/rsna_cache --epochs 8`.
  - Save Top-3 checkpoint average (`dinov2_slothead_top3.pth`) and verify overall notebook runtime $\le 35$ minutes.
- [ ] **Verification**:
  - Unit tests verify 5-slice cache reading, dynamic 3-slice window slicing, and augmentation symmetry constraints.
  - Synthetic DDP smoke test verifies parameter group transitions across the 3 unfreezing stages without CUDA OOM or DDP synchronization stalls.
