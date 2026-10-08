# Five-Slice Pre-Caching with Dynamic 2.5D Depth Jitter and Progressive ViT Unfreezing

We standardized on pre-caching 5 consecutive physical slices per contrast slot in session scratch (`/kaggle/tmp/rsna_cache`) with runtime dynamic 3-slice window sampling, combined with a 3-stage progressive vision transformer unfreezing schedule and symmetry-preserving MSK augmentations.

## Status

accepted

## Context

In our initial Dual-T4 DistributedDataParallel training run (`outputs/kaggle_runs/rsna-efficiency-track-ddp-training`), training took 8 hours and 21 minutes (30,070s) across 12 epochs (~40 minutes/epoch) and exhibited severe validation overfitting starting after Epoch 7:
1. **I/O Bottleneck**: Because `--cache-dir` was not specified, the DataLoader parsed and decoded raw DICOM pixel arrays for 4,407 studies on every single epoch (52,884 total decodings), leaving dual T4 GPUs idling while waiting for CPU disk I/O.
2. **Deterministic Input Memorization**: The model saw identical static center crops (`slice_pos = 0.5`) with zero spatial or photometric augmentations.
3. **Premature ViT Destruction**: The 22M-parameter DINOv2-Small backbone was unfrozen from step 1 at $\text{LR}=3\times 10^{-5}$, enabling rapid memorization of the 2,938 un-augmented training studies.

## Considered Options

- **Static 3-Slice Pre-Caching**: Pre-caching only 3 center slices (`shape (6, 3, 336, 336)`). Eliminates I/O bottleneck but permanently locks depth position, preventing volumetric slice jitter.
- **Online DICOM Depth Jitter**: Randomizing `slice_pos` during raw DICOM decoding on every batch. Provides volumetric variance but retains the ~40 min/epoch CPU decoding bottleneck.
- **Five-Slice Caching + Dynamic 3-Slice Window (Selected)**: Pre-cache 5 consecutive physical slices centered on joint normal ($[c-2 \dots c+2]$, shape `(6, 5, 336, 336)` float16) during a one-time ~16-minute preprocessing pass. During training, DataLoader randomly slices a 3-slice window (`[0:3]`, `[1:4]`, or `[2:5]`), providing dynamic volumetric depth jitter with zero DICOM decoding during training.
- **Single-Stage Full Backbone Fine-Tuning**: Training all 12 ViT blocks from Epoch 1. Causes early overfitting.
- **3-Stage Progressive Unfreezing (Selected)**:
  - *Stage 1 (Epochs 1–2)*: Freeze backbone; train SlotHead queries, cross-attention, and classifier head at $\text{LR}=1\times 10^{-3}$.
  - *Stage 2 (Epochs 3–4)*: Unfreeze top 2 transformer blocks (`blocks[10:]` + `norm`) at low $\text{LR}=1\times 10^{-5}$.
  - *Stage 3 (Epochs 5–8)*: Unfreeze all 12 blocks with differential learning rate decaying to zero.

## Consequences

- Total 8-epoch training run completes in $\le 35$ minutes on Kaggle Dual-T4 GPUs (16m one-time preprocessing + 16m training).
- Cache footprint in `/kaggle/tmp/rsna_cache` is ~15 GB (within Kaggle scratch disk limits).
- Dynamic depth jitter combined with symmetry-preserving MSK augmentations (rotation $\pm 7^\circ$, scale $0.93\text{--}1.07$, strictly no horizontal flips) eliminates static input memorization.
- Top-3 checkpoint averaging (`dinov2_slothead_top3.pth`) captures peak generalization without post-epoch-7 divergence.
