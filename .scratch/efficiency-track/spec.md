# RSNA Knee Abnormality Detection: Efficiency Track Specification (DINOv2-Small 6-Slot SlotHead)

**Status:** ready-for-agent
**Target:** Kaggle RSNA Efficiency Track ($10,000+ Prize Category) & Standalone High-Speed Submission
**Compute Target:** Dual Tesla T4 GPUs (Kaggle Environment), total test set inference ≤ 15 minutes

---

## 1. Problem Statement

While top ensemble solutions on the Main Track leaderboard achieve ~0.940 Macro ROC-AUC by blending up to 39 model checkpoints across 4 foundation arms, they require 8.5 to 8.9 hours of execution on dual T4 GPUs. This heavy architecture is ineligible for the Efficiency Track, risks Kaggle 9-hour session timeout failures, and suffers from massive I/O serialization bottlenecks during slice ordering.

Furthermore, naive attempts to utilize Kaggle's dual T4 setup via `nn.DataParallel` cause catastrophic host RAM memory leaks and 7× batch serialization slowdowns ([ADR-0003](file:///home/wenalz/Documents/Antigravity%20projects/RSNA_MRI/docs/adr/0003-single-gpu-execution-over-dataparallel.md)).

Our current baseline (v16 EfficientNet-B2 at 0.8008 Macro AUC) lacks physical coordinate sorting, physical millimeter field-of-view (FOV) normalization, and multi-sequence cross-attention, leaving substantial performance and efficiency on the table.

---

## 2. Solution Overview

A self-contained, lightning-fast Efficiency Track pipeline featuring:
1. **DINOv2-Small Backbone (`vit_small_patch14_dinov2`)**: 336×336 px resolution, capturing rich self-supervised anatomical and soft-tissue representations.
2. **6-Slot SlotHead Cross-Attention**: 6 anatomical contrast slots with 3 adjacent slices ($\pm 1$) per slot (18 slices total per study). A learnable `(12, d)` query matrix attends over slot representations, allowing finding-specific sequence routing.
3. **Optimized DICOM Geometry Pipeline**: Physical normal plane projection sorting ($k = \vec{p} \cdot (\vec{r} \times \vec{c})$) with `stop_before_pixels=True` in a 16-worker thread pool, followed by a fixed 140 mm physical FOV crop via `PixelSpacing`.
4. **Leak-Free Dual-T4 Acceleration**:
   - **Training**: DistributedDataParallel (DDP) via `torchrun` / `torch.multiprocessing.spawn`, avoiding host RAM scatter/gather overhead.
   - **Inference**: Parallel process worker pool distributing studies 50/50 across `cuda:0` and `cuda:1`.
5. **Full TensorBoard Diagnostic Suite**: Computational model graph, train/val losses, learning rate schedules, macro ROC-AUC, 12 per-label ROC-AUC curves (with Synovitis evaluated strictly on Gold Studies), finding-to-slot attention heatmaps, and PR curves.
6. **Calibrated Weak Supervision**: Trained against Gemini Phase 2 calibrated soft probabilities (`labels_soft.parquet`) with square-root damped `pos_weights` across a 3-Fold Scanner-Stratified GroupKFold split.

---

## 3. User Stories

1. **As an Efficiency Competitor**, I want the test inference pipeline to process all ~1,300 test studies in ≤ 15 minutes on dual T4s so that our solution places in the top echelon of the Efficiency Track leaderboard.
2. **As an ML Engineer**, I want both Kaggle T4 GPUs utilized simultaneously via DDP during training and multiprocessing during inference without host RAM crashes or PCIe bottlenecks.
3. **As an Experimenter**, I want full TensorBoard integration logging model architecture graphs, per-label AUCs, loss curves, and slot attention maps to track training dynamics in real time.
4. **As an Anatomical Modeler**, I want DICOM slices sorted strictly by physical normal projection and cropped to a 140 mm physical box so that the network receives vendor-invariant, anatomically aligned knee views.
5. **As an Evaluator**, I want the model to output raw logits of shape `(B, 12)` and compute validation AUC with gold-filtered Synovitis evaluation to maintain strict concordance with the competition metric.

---

## 4. Implementation Decisions

### 4.1 DICOM Preprocessing & Geometry Pipeline (`src/datasets/`)
- **Normal Projection Sorting**:
  ```python
  n = np.cross(iop[:3], iop[3:])  # normal = row_cosines x col_cosines
  k = float(np.dot(ipp, n))       # projection of position onto normal
  ```
- **Fast Header Scanning**: Read DICOM headers with `pydicom.dcmread(..., stop_before_pixels=True)` inside `concurrent.futures.ThreadPoolExecutor(max_workers=16)` to drop slice ordering time from 12 minutes to under 45 seconds.
- **Physical Millimeter Crop**:
  ```python
  crop_pixels = int(round(140.0 / max(pixel_spacing, 1e-3)))
  ```
  Centered on the joint array, clipped to image bounds, and resized to 336×336 using `cv2.INTER_AREA`.
- **6 Contrast Slot Layout**:
  - Slot 0: `SAG_FLUID_FS` (Sagittal, Fluid-sensitive, Fat-suppressed)
  - Slot 1: `COR_FLUID_FS` (Coronal, Fluid-sensitive, Fat-suppressed)
  - Slot 2: `AX_FLUID_FS` (Axial, Fluid-sensitive, Fat-suppressed)
  - Slot 3: `SAG_FLUID_NOFS` (Sagittal, Fluid-sensitive, Non-fat-suppressed)
  - Slot 4: `COR_T1` (Coronal, T1-weighted structural)
  - Slot 5: `SAG_T1` (Sagittal, T1-weighted structural)
- **Slice Triplets (2.5D)**: For each slot, sample the central anatomical slice $c$ and its immediate neighbors $[c-1, c, c+1]$ stacked into 3 channels. If a sequence is absent, pass a zero-filled tensor and mark presence mask as 0.

### 4.2 Architecture: DINOv2-Small SlotHead (`src/models/dinov2_slothead.py`)
- **Encoder**: `timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)` (embedding dim $D=384$, input size 336×336).
- **SlotHead Cross-Attention Module**:
  - Input: 6 slot image triplets $\to$ zero-gated for absent sequences $\to$ forward through DINOv2 $\to$ slot CLS tokens $S \in \mathbb{R}^{B \times 6 \times D}$.
  - Slot Identity Embeddings: $E_{\text{slot}} \in \mathbb{R}^{6 \times D}$ injected into $S$ so attention keys possess explicit sequence identity.
  - Learnable Queries: $Q \in \mathbb{R}^{12 \times D}$ representing the 12 clinical knee findings.
  - Multi-Head Cross-Attention ($H=6$ heads): $Q$ attends to $(S + E_{\text{slot}})$ with `key_padding_mask` gating absent slots; all-absent studies unmasked to zero-tokens to prevent NaN softmax.
  - Residual & LayerNorm: $A = \text{LayerNorm}(Q + \text{Attn}(Q, S + E_{\text{slot}}))$.
  - Per-Finding Classifier Head: $W_{\text{cls}} \in \mathbb{R}^{12 \times D}$ and $b_{\text{cls}} \in \mathbb{R}^{12}$, computing $\text{logits}_f = \langle A_f, W_f \rangle + b_f$, preventing logit collapse across findings.
- **Model Output**: Strictly raw logits of shape `(batch_size, 12)`. Sigmoid applied only during evaluation/inference.

### 4.3 Dual-T4 Distributed Execution (`src/training/` & `notebooks/`)
- **Training (DDP)**:
  - Spawn 2 processes (one per T4: `local_rank` 0 and 1).
  - Wrap model in `torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])`.
  - Use `DistributedSampler` to shard the 4,407 training studies.
  - Gradients all-reduced cleanly across internal NVLink/PCIe without passing high-dimensional MRI tensors through CPU memory.
- **Inference Sharding**:
  - Split test study IDs into 2 equal lists.
  - Launch 2 subprocesses (`cuda:0` and `cuda:1`) via `concurrent.futures.ProcessPoolExecutor(max_workers=2)`.
  - Concatenate resulting prediction dataframes into final `submission.csv`.
- **Runtime Target**: 1,300 test studies evaluated in ~10–14 minutes total.

### 4.4 Full TensorBoard Integration (`src/utils/tensorboard.py`)
- **Writer Directory**: `outputs/tensorboard/<experiment_tag>/`
- **Model Graph**: Logged on step 0 via `writer.add_graph(model, dummy_input)`.
- **Scalars Tracked**:
  - `Loss/train_step`, `Loss/train_epoch`, `Loss/val_epoch`
  - `LR/backbone`, `LR/slothead`
  - `AUC/macro_val`
  - `AUC_Per_Label/<Finding_Name>` (12 individual scalar curves)
- **Slot Attention Visualization**:
  - At each validation epoch, compute the $12 \times 6$ attention matrix (12 findings $\times$ 6 slots) on a fixed cohort of Gold Studies.
  - Render via `matplotlib` heatmap and log using `writer.add_figure("Attention/Finding_to_Slot", fig, epoch)`.
- **Precision-Recall Curves**:
  - `writer.add_pr_curve(f"PR/{Finding_Name}", labels, probs, epoch)` for all 12 findings.

### 4.5 Training Recipe & Optimization
- **Loss**: `MaskedBCEWithLogitsLoss` supporting continuous soft targets, NaN masking for acute tear silence, and continuous positive class weighting ($w_f = 1.0 + (\text{pos\_weight}_f - 1.0) \cdot y_f$):
  `pos_weight = [2.70, 6.98, 1.21, 2.34, 2.12, 3.01, 1.83, 1.60, 2.94, 2.49, 2.21, 3.55]`
- **Optimizer**: AdamW with layer-wise differential learning rates and weight decay exclusions:
  - Backbone: `3e-5` (decay: `0.02` on 2D weights; `0.0` on 1D biases, norms, position tokens)
  - SlotHead & Classifier: `1e-3` (decay: `0.02` on 2D weights; `0.0` on biases, LayerNorm, query/slot tokens)
- **Scheduler**: `OneCycleLR` (12 epochs, `pct_start=0.15`, cosine annealing).
- **Precision & Backend**:
  - `torch.backends.cudnn.benchmark = True`
  - `torch.autocast("cuda", dtype=torch.float16)` with automatic full-precision FP32 fallback for attention stability.
- **Checkpoint Averaging**: Top-3 best validation epoch weights averaged post-training.

---

## 5. Testing Decisions

1. **Geometry Unit Tests (`tests/test_efficiency_geometry.py`)**:
   - Assert physical normal projection produces monotonic spatial ordering across multi-vendor test series.
   - Assert millimeter cropping produces identical millimeter joint bounding boxes regardless of input pixel spacing (0.3 mm vs 0.6 mm).
2. **Model Architecture Unit Tests (`tests/test_dinov2_slothead.py`)**:
   - Verify input shape `(B, 6, 3, 336, 336)` returns output logits of shape `(B, 12)`.
   - Verify presence mask zero-gating functions correctly when slots are missing.
   - Assert attention weights sum to 1.0 across the 6 slots for each finding.
3. **TensorBoard Integration Tests (`tests/test_tensorboard_logging.py`)**:
   - Smoke test `TensorBoardLogger` to verify scalar, figure, and graph events write cleanly without throwing serialization errors.
4. **Dual-GPU DDP Smoke Test (`scripts/smoke_ddp.py`)**:
   - Verify 2-worker DDP initializes, executes 2 training steps, and synchronizes weights without hanging or host memory growth.

---

## 6. Out of Scope

- **Main Track 39-Model Ensemble**: Defer all multi-arm blending (CoAtNet + RadImageNet + DINOv3) to Main Track development.
- **Dense Sliding Window Overlap (94 Windows)**: Dropped due to 9-hour inference requirement.
- **Random Cross-Validation Splits**: Strictly prohibited due to scanner signature leakage (+0.053 artificial AUC).
- **Hard Label Binarization**: Rejected; continuous calibrated soft labels are required to prevent loss penalties on hedged radiologist sentences.
- **Softmax Activation**: Prohibited; each finding is an independent binary classification task.

---

## 7. Performance Benchmarks

| Metric | Current Baseline (v16) | Efficiency Track Target |
|---|---|---|
| **Architecture** | EfficientNet-B2 (Single) | DINOv2-Small (6-Slot SlotHead) |
| **Validation Macro ROC-AUC** | 0.8008 ± 0.0042 | **≥ 0.900 – 0.915** |
| **Test Set Inference Time** | ~45 min (Single T4) | **≤ 15 min (Dual T4)** |
| **GPU Utilization** | 1× T4 (50% system idle) | **2× T4 (100% active via DDP/sharding)** |
| **Slice Ordering Time** | ~12 min | **< 45 sec (16-thread pool)** |
| **Efficiency Leaderboard Tier** | Uncompetitive | **Top 5 – Top 10 Contender** |

---

## 8. Cross References

- **Ubiquitous Domain Model**: [`CONTEXT.md`](file:///home/wenalz/Documents/Antigravity%20projects/RSNA_MRI/CONTEXT.md)
- **ADR-0001**: [Two-Tier LLM Weak Label Extraction](file:///home/wenalz/Documents/Antigravity%20projects/RSNA_MRI/docs/adr/0001-two-tier-llm-weak-label-extraction-and-imputation.md)
- **ADR-0002**: [2.5D Stacking & Attention Pooling](file:///home/wenalz/Documents/Antigravity%20projects/RSNA_MRI/docs/adr/0002-2.5d-mil-with-attention-pooling.md)
- **ADR-0003**: [Single-GPU vs Multi-GPU Execution](file:///home/wenalz/Documents/Antigravity%20projects/RSNA_MRI/docs/adr/0003-single-gpu-execution-over-dataparallel.md)
- **ADR-0005**: [3-Fold Stratified GroupKFold Standardization](file:///home/wenalz/Documents/Antigravity%20projects/RSNA_MRI/docs/adr/0005-three-fold-stratified-groupkfold-standardization.md)
- **Competitor Teardown**: [`search/public_notebooks_research/01_FOUR_ARM_ARCHITECTURE_TEARDOWN.md`](file:///home/wenalz/Documents/Antigravity%20projects/RSNA_MRI/search/public_notebooks_research/01_FOUR_ARM_ARCHITECTURE_TEARDOWN.md)
- **Speed Optimization Guide**: [`search/public_notebooks_research/05_EFFICIENCY_TRACK_AND_SPEED_OPTIMIZATION.md`](file:///home/wenalz/Documents/Antigravity%20projects/RSNA_MRI/search/public_notebooks_research/05_EFFICIENCY_TRACK_AND_SPEED_OPTIMIZATION.md)
