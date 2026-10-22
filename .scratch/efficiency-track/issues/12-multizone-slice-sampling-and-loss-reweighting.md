# 12: Multi-Zone Slice Sampling (3 Compartments / 9 Slices), 18-Token SlotHead & Per-Label Loss Reweighting

**What to build:** Upgrade the DINOv2-Small SlotHead DDP training pipeline from 5 central slices to a 9-slice 3-zone compartment-anchored sampling geometry (capturing peripheral meniscal horns and articular cartilage margins at 25%, 50%, and 75% normalized depth), expand SlotHead cross-attention to 18 slot-zone tokens with additive contrast and depth embeddings, implement normalized per-label task loss reweighting ($1.75\times$ target boost on `Medial Meniscus`, `Lateral Meniscus`, `Lateral OA`, and `PF OA`), configure Dual-T4 DDP for safe execution with batch size 4 and gradient accumulation 2, update unit tests and the Kaggle training notebook generator, and deploy the updated codebase and kernel to Kaggle.

**Blocked by:** 11: Multi-Fold Pipeline Runner (Folds 1 & 2), Percentile Rank OOF Evaluation & Single-Model Refinement

**Status:** completed

---

## Critical Directive: Diff-Not-Overwrite Execution Rule

> [!CAUTION]
> **DO NOT paste entire files over existing repository files.**
> All changes MUST be applied as targeted diffs against HEAD (`git diff` review before commit).
> Any file or function not explicitly enumerated below MUST remain untouched.

### File & Function Scope Classification:
- **New Code**:
  - `src/utils/seed.py`: `seed_everything` standalone helper (breaks circular dependencies).
  - `src/datasets/efficiency_pipeline.py`: `select_multizone_slice_paths` (with $\pm 3.0$ mm physical check for thin 3D series), `process_study_to_9slice_cache`, 9-slice branch of `verify_cache`.
  - `scripts/scratch_time_inference.py`: Single-GPU 100-study inference benchmark script (Gate 2).
  - `tests/test_multizone_sampling.py`: Comprehensive test suite for multi-zone sampling, 18-token slothead, task loss reweighting, fail-fast cache invariants, and DDP unfreezing.
- **Changed Code (Targeted Diffs Only)**:
  - `src/datasets/efficiency_pipeline.py`: `normalize_volume_dinov2` ($K=9$ branch only), `verify_cache` (enforce `.npz` only, strict format key check).
  - `src/models/dinov2_slothead.py`: `DINOv2SlotHead` (18-token cross-attention with `zone_embed`, 3-zone cluster masking, slot dropout, LayerNorm(Q+Attn) residual, pre-classifier dropout, unmasking for all-absent studies).
  - `src/training/losses.py`: `compute_task_weights` and `task_weights` buffer support in `MaskedBCEWithLogitsLoss` (preserving class default `gold_weight: float = 2.0` and `continuous_pos_weight: bool = True`).
  - `src/training/train_efficiency.py`: `impute_soft_synovitis` (safe fillna & pre-binarized silence handling), `apply_msk_augmentations` (per-slot 9-slice affine with $\pm 4\%$ translation and corner-median fill), `EfficiencyStudyDataset` (fail-fast `.npz` only, default transform on `is_train`), `build_differential_optimizer` (`no_decay` for `pos_embed`, `cls_token`, and `zone_embed`, explicit `family` tags in groups).
  - `scripts/train_ddp_efficiency.py`: `contextlib.nullcontext()` for `sync_context`, `--limit` flag with cached UID filtering before `verify_cache`, timers for s/step / data_wait / peak VRAM, NaN AUC tolerance for smoke runs, `--force-retrain`, `--gold-weight None` with fallback to `config.yaml`, robust label merge (`validate="one_to_one"`).
  - `scripts/preprocess_336px_cache.py`: 9-slice pre-caching with `--benchmark` and `benchmark_loader_throughput` restored, pilot routing, resume shape validation, QC montage on zone 1 centre slice (index 4).
  - `src/utils/tensorboard.py`: `import math` added, `compute_conditional_slot_attention` patched to aggregate 18 tokens into 12×6.
  - `scripts/make_training_notebook.py`: Notebook generator updated for 9-slice cache path (~30 GB disk assert `free_gb >= 45.0`), batch size 4, grad accum 2, boost 1.75.
- **Unchanged Code (Must Remain Strictly Intact)**:
  - All legacy 3- and 5-slice helpers in `src/datasets/efficiency_pipeline.py` (`process_study_to_6slot`, `process_study_to_5slice_cache`, `Fast6SlotDICOMDataset`, `select_triplet_paths`, `select_5slice_paths`, `decode_triplet_slices`, `normalize_triplet_dinov2`, `CropMask`, `crop_and_resize_140mm`).
  - `compute_collective_val_loss` and `PlattScaler` in `src/training/train_efficiency.py` and `src/training/losses.py`.
  - `src/training/ddp.py` (`set_stage_trainable` freezes only backbone; all head parameters including `zone_embed` train from Stage 1 through Stage 4).

### Intentionally Deleted or Rewritten Tests:
The 5-slice dataset tests in `tests/test_caching_and_regularization.py` (`test_dynamic_depth_jitter_offset_coverage`, `test_train_val_inference_parity` eval window `[1:4]`, and `test_efficiency_dataset_npz_loading` with shape `(6, 5, 336, 336)`) test 5-slice dynamic depth jitter behavior that Issue 12 deliberately replaces with fixed 9-slice multi-zone caching. They are superseded by ADR-0010 (superseding ADR-0006's depth jitter) and tested via `tests/test_multizone_sampling.py`. During implementation, these tests are rewritten for 9 slices or retired.

---

## Four Operational Execution Gates

1. **Gate 1: Pilot Smoke Run & Full Budget Projection (50 Studies)**
   - Pre-cache 50 studies via:
     `python scripts/preprocess_336px_cache.py --dicom-root <DICOM_ROOT> --series-csv data/raw/train_series.csv --splits-csv data/splits/cv_folds_3fold.csv --limit 50 --cache-dir /tmp/rsna_pilot_cache --num-workers 4 --benchmark`
   - Run 14 epochs of DDP training on 50 studies to fully exercise Stage 4 peak VRAM:
     `python scripts/train_ddp_efficiency.py --folds 0 --limit 50 --epochs 14 --batch-size 4 --grad-accum-steps 2 --cache-dir /tmp/rsna_pilot_cache --weights-path <BACKBONE.pth> --labels-path data/labels/pseudo_labels.csv --splits-path data/splits/cv_folds_3fold.csv --weights-dir /tmp/pilot_weights --tb-dir /tmp/pilot_tb --force-retrain`
     *(Note: Invoke directly via `python scripts/train_ddp_efficiency.py`; the runner manages `mp.spawn` per GPU internally.)*
   - Script logs: `s/micro-batch`, `data_wait` seconds, and `peak VRAM GiB` (`torch.cuda.max_memory_allocated(device)/2**30`). Judge `data_wait` on the first epoch to detect cold disk I/O bottlenecks.
   - Verify Cache Disk Footprint: Multiply measured bytes/study by 4,407 and assert projected total disk usage is $< 45\text{ GB}$.
   - Project wall-clock budget for full Kaggle session:
     $$T = T_{\text{cache}} + 3 \cdot \sum_{\text{stage}} E_{\text{stage}} \cdot (t_{\text{micro,stage}} \cdot M_{\text{train}} + t_{\text{val}}) + T_{\text{oof}} + 0.25\text{ h}$$
     where $E = (2, 2, 3, 7)$, $M_{\text{train}} = \lceil N_{\text{train\_fold}} / (2\text{ GPUs} \cdot 4\text{ batch}) \rceil \approx 368$, $T_{\text{cache}} = \frac{\text{cache\_s\_per\_study} \times 4407}{3600}$, $t_{\text{val}}$ is measured per epoch, and $T_{\text{oof}} \approx 0.10\text{ h}$.
   - **Pass Criterion**: $T \le 8.0\text{ h}$ (strictly within Kaggle's 9.0 h hard ceiling) and Stage 4 peak VRAM $< 14.0\text{ GiB}$.
   - **Mandatory Stop & Report**: The executing agent must stop and report Gate 1 performance numbers before proceeding.

2. **Gate 2: Inference Latency Timing Benchmark**
   - Execute `scripts/scratch_time_inference.py --dicom-root data/train --splits-path data/splits/cv_folds_3fold.csv --n-studies 100 --n-models 1 --device cuda:0`.
   - Script evaluates online DICOM extraction with `num_workers=2` (reflecting dual GPU process contention across 4 vCPUs) and ViT forward passes for $n_{\text{models}} \in \{1, 3\}$.
   - Calculates projected test time:
     $$\text{Governing Min} = \max\left(\frac{650 \cdot (t_{\text{decode}} + n_{\text{models}} \cdot t_{\text{forward}})}{60.0}, \frac{650 \cdot t_{\text{decode}}}{60.0}\right)$$
   - **Pass Criterion**: Governing runtime $\le 15.0\text{ minutes}$ (900.0 seconds).

3. **Gate 3: Target Loss Boost Ablation (Fold 0)**
   - Train Fold 0 with uniform weights (`--target-label-boost 1.0`) to `outputs/ablation_boost_1.0`.
   - Train Fold 0 with boosted weights (`--target-label-boost 1.75`) to `outputs/ablation_boost_1.75` (supported by `--force-retrain`).
   - Invoke directly via `python scripts/train_ddp_efficiency.py` passing required paths.
   - **Pass Criterion**: Boosted run demonstrates improved Macro AUC on peripheral targets (`Medial Meniscus`, `Lateral Meniscus`, `Lateral OA`, `PF OA`) without degrading Notch/Fluid targets (`ACL`, `Effusion`).

4. **Gate 4: Preflight Tooling & Regression Verification**
   - Verify `--help` across all CLI scripts (`preprocess_336px_cache.py`, `train_ddp_efficiency.py`, `evaluate_3fold_oof.py`, `make_training_notebook.py`).
   - Confirm `evaluate_3fold_oof.py` accepts all flags passed in notebook Cell 6 (`--weights-dir`, `--splits-path`, `--labels-path`, `--config-path`, `--cache-dir`, `--output-csv`, `--output-json`).
   - Run `pytest -x` ensuring all unit tests pass with exit 0.

---

## Tasks

- [x] **3-Zone Compartment-Anchored Slice Sampling (`src/datasets/efficiency_pipeline.py`)**:
  - Implement `select_multizone_slice_paths(headers, num_zones=3)`:
    - Sort deduplicated headers along normal projection vector $k = \vec{p} \cdot (\vec{r} \times \vec{c})$.
    - Check median slice spacing $\Delta k$: if $\Delta k < 1.5$ mm (thin-slice 3D series), select physical slices at $\pm 3.0$ mm around zone centers. Otherwise select index offsets $[c_z - 1, c_z, c_z + 1]$.
    - Anchor 3 zone centers at normalized positions: $c_0 = \text{round}(0.25 \times (N - 1))$, $c_1 = \text{round}(0.50 \times (N - 1))$, $c_2 = \text{round}(0.75 \times (N - 1))$.
    - Handle short series ($N < 9$) with `np.clip(idx, 0, N - 1)` boundary clamping.
  - Implement `process_study_to_9slice_cache`:
    - Output study tensor of shape `(6, 9, 336, 336)` float16 and binary presence mask `(6,)`.
  - Update `normalize_volume_dinov2` to handle $K=9$ volumes.
  - Update `verify_cache` to strictly require `.npz` format, validate `format` key, `n_slices_cached: 9`, and `sampling_strategy: "3zone"`.
- [x] **Dedicated 9-Slice Offline Pre-Caching Pipeline (`scripts/preprocess_336px_cache.py`)**:
  - Update pre-caching script to extract and persist 9 slices per slot into compressed `.npz` archives.
  - Default `--cache-dir` to `/kaggle/tmp/rsna_cache_336_9slice`.
  - Update `cache_manifest.json` schema: `"format": "npz"`, `"n_slices_cached": 9`, `"sampling_strategy": "3zone"`.
  - Update resume logic to verify existing `.npz` shape is `(6, 9)`.
  - Update QC montage to select zone 1 centre slice (index 4).
  - Restore `--benchmark` CLI flag and `benchmark_loader_throughput`.
- [x] **18-Token Multi-Zone DINOv2-Small SlotHead (`src/models/dinov2_slothead.py`)**:
  - Add learnable depth zone embeddings `self.zone_embed = nn.Parameter(torch.randn(3, embed_dim) * 0.20)`.
  - Reshape input `x` of shape `(B, 6, 9, H, W)` into 18 triplets `(B, 6, 3, 3, H, W)`.
  - Forward active slot-zone triplets through shared DINOv2-Small backbone under non-reentrant gradient checkpointing.
  - Construct 18 cross-attention tokens with additive embeddings: $\mathbf{t}_{s, z} = \mathbf{f}_{s, z} + \mathbf{e}_{\text{slot}, s} + \mathbf{e}_{\text{zone}, z}$.
  - Update `key_padding_mask` so absent slots mask all 3 corresponding zone tokens simultaneously.
  - Unmask `key_padding_mask` for all-absent studies to prevent NaN softmax and logits.
  - Preserve LayerNorm residual `self.norm(q + attn_out)` and pre-classifier dropout `self.pre_cls_dropout(a)`.
  - Default `num_heads` to 6.
  - Cross-attention queries $\mathbf{Q} \in \mathbb{R}^{12 \times D}$ attend across all 18 tokens, returning raw logits of shape `(B, 12)`.
  - Preserve `forward_with_attention` returning `(logits, attn_weights)`.
- [x] **Normalized Task-Level Loss Reweighting (`src/training/losses.py`)**:
  - Add `compute_task_weights(target_boost: float = 1.75, boost_indices=(2, 3, 5, 6))` returning normalized weights with mean 1.0 ($1.40\times$ for the 4 boosted targets, $0.80\times$ for remaining 8).
  - Add `task_weights: torch.Tensor | None = None` parameter of shape `(12,)` to `MaskedBCEWithLogitsLoss`. Preserves `gold_weight: float = 2.0` class default and `continuous_pos_weight: bool = True`.
  - Multiply unreduced BCE loss by `task_weights` buffer placed on compute device alongside `sample_weights` and continuous `pos_weight`.
- [x] **DDP Training Engine & CLI Wiring (`src/training/train_efficiency.py` & `scripts/train_ddp_efficiency.py`)**:
  - In `EfficiencyStudyDataset`, require mandatory 9-slice `.npz` cache and fail fast on missing manifest, shape mismatch, or missing files. Default `self.transform` to `apply_msk_augmentations` when `is_train=True`.
  - Full MSK augmentations: per-slot affine (rot $\pm 7^\circ$, scale $0.93\text{--}1.07$, translate $\pm 4\%$), corner-median background fill, and global contrast $\pm 10\%$.
  - Synovitis imputation: `impute_soft_synovitis` fills both `isna()` and pre-binarized `(Synovitis == 0.0) & synovitis_soft.between(0.001, 0.999)` for non-gold studies (3,540 studies), preserving gold hard labels. Runtime guard verifies `n_imputed > 0 or n_frac > 0`.
  - Optimizer: add `"pos_embed"`, `"cls_token"`, and `"zone_embed"` to `no_decay`, add explicit `"family"` key to every group, and assert complete parameter coverage.
  - In `train_ddp_efficiency.py`:
    - Use `group["family"]` for family validation and LR scheduling.
    - `sync_context = contextlib.nullcontext() if is_accum_boundary else ddp_model.no_sync()`.
    - Drop unused `rewrap_ddp` import and remove duplicate `--force-retrain`.
    - Add `--limit` flag with cached UID filtering before `verify_cache`.
    - Epoch timers: `s/step`, `data_wait`, `peak VRAM GiB`.
    - Set `--gold-weight None` by default, resolving to `config.yaml` (`training.gold_weight`).
    - Seed hygiene: `seed_everything` before model initialization, followed by per-rank seed.
    - Robust label merge with `validate="one_to_one"` and count assertion.
- [x] **Telemetry Suite (`src/utils/tensorboard.py`)**:
  - Add `import math` to prevent `NameError` in `log_scalar`.
  - Update `compute_conditional_slot_attention` to aggregate 18 tokens into 12×6 slot attention matrix (`matrix.reshape(b, f, num_slots, t // num_slots).sum(axis=-1)`).
- [x] **Timing Benchmark Script (`scripts/scratch_time_inference.py`)**:
  - Implement single-GPU 100-study timing benchmark evaluating DICOM decoding + forward pass with untimed warm-up and FP16 autocast, reporting dual projections against 15-minute budget.
- [x] **Comprehensive Unit Test Suite (`tests/test_multizone_sampling.py`)**:
  - Test `test_select_multizone_slice_paths_spacing_and_anchors` ($N=25$ hardcoded [6, 12, 18], thin 3D $\pm 3.0$ mm spacing [54, 60, 66]).
  - Test `test_all_absent_study_produces_finite_logits`.
  - Test `test_head_incl_zone_embed_trains_from_stage1_and_all_at_stage4` (verifies `zone_embed` trains under `set_stage_trainable`).
  - Test `test_soft_synovitis_imputation_and_gold_protection`.
  - Test `test_apply_msk_augmentations_shape_and_absent_slot_untouched`.
  - Test `test_tensorboard_logger_smoke_scalar` with `TensorBoardLogger(log_dir=tmp_path)`.
  - Test `test_efficiency_study_dataset_rejects_legacy_5slice_npz` and `test_efficiency_study_dataset_raises_on_missing_npz`.
  - Test closed-form task loss gradients and optimizer family coverage.
- [x] **Notebook Synchronization & Kaggle Deployment (`scripts/make_training_notebook.py`)**:
  - Update pre-caching command to `/kaggle/tmp/rsna_cache_336_9slice` with `assert free_gb >= 45.0`.
  - Update DDP command to `--batch-size 4 --grad-accum-steps 2 --target-label-boost 1.75`.
  - Regenerate `notebooks/rsna_efficiency_training.ipynb`.
