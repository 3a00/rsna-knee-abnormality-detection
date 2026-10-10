# 336px Nyquist Sampling, 140mm Physical FOV Cropping, and Calibrated Soft Weak Supervision for ViT

We adopt 336×336 px input resolution with 140 mm physical FOV joint centering, dedicated offline pre-caching, continuous calibrated soft weak supervision across all pseudo-label studies, a $2.0\times$ gold study loss weighting multiplier, and an extended 14-epoch 4-stage progressive unfreezing schedule to push the single DINOv2-Small SlotHead model toward 0.90+ Macro AUC.

## Status

accepted (extends ADR-0007)

## Context

In our Phase 3 v17 / Efficiency Track DDP baseline, DINOv2-Small reached 0.8362 Macro AUC on 224×224 px inputs over 10 epochs. Analysis of the logs, validation curves, and competitive research identified three fundamental bottlenecks limiting further gains:
1. **The Nyquist Tear Resolution Dilemma**: At 224×224 px without physical cropping, a 14px DINOv2 patch token spans ~16.7 mm, meaning focal 1–3 mm meniscus and cartilage tears become sub-pixel artifacts that cannot be resolved by ViT self-attention.
2. **Under-Trained Stage 4 Convergence**: Validation loss was at its global minimum (0.6042) at Epoch 10 with zero signs of overfitting, indicating the model was truncated before full convergence of the unfrozen lower transformer layers.
3. **Hard Binarization Gradient Discontinuity**: Training on hard 0/1 binarized labels discarded subtle probability gradations and clinical hedging captured by Gemini extraction, causing sharp gradients on borderline pathologies.

## Considered Options

- **Retain 224×224 Resolution with Online Upsampling**: Avoids re-caching but fails to recover high-frequency spatial detail required for articular cartilage and meniscus margin tears.
- **336×336 px Physical 140 mm Joint Centering (Selected)**:
  - Crop `round(140.0 / PixelSpacing)` pixels centered on the knee joint before resizing to 336×336 px.
  - Reduces the physical patch token footprint from ~16.7 mm down to ~5.8 mm, satisfying the Nyquist sampling criterion for focal tears.
  - Interpolate DINOv2 positional embeddings bicubically from $(37, 37)$ down to $(24, 24)$.
- **Dedicated Offline Pre-Caching (`scripts/preprocess_336px_cache.py`) (Selected)**:
  - Pre-crop and cache 5-slice stacks ($[c-2 \dots c+2]$) across the 6 anatomical slots into compressed `.npz` files (~18 GB dataset).
  - Eliminates online multi-frame DICOM CPU decoding bottlenecks during Dual-T4 DDP training, keeping epoch duration under ~3.5 minutes.
- **Calibrated Continuous Soft Supervision & $2.0\times$ Gold Multiplier (Selected)**:
  - Supervise pseudo-label studies with continuous Gemini posterior probabilities $[0.0, 1.0]$ in `BCEWithLogitsLoss`.
  - Apply a $2.0\times$ sample loss multiplier to the 58 gold ground-truth studies to anchor decision boundaries to authoritative radiologist consensus.
- **14-Epoch 4-Stage Progressive Unfreezing Schedule (Selected)**:
  - *Stage 1 (Epochs 1–2)*: Classifier head & SlotHead warmup.
  - *Stage 2 (Epochs 3–4)*: Top-2 blocks unfreezing (`blocks[10:]`).
  - *Stage 3 (Epochs 5–7)*: Top-4 blocks unfreezing (`blocks[8:]`).
  - *Stage 4 (Epochs 8–14)*: All 12 blocks unfreezing with 4-tier LLRD and cosine decay to zero over 7 full epochs.
- **Stage-4 Top-3 Weight Averaging (Selected)**:
  - Linearly average the state dictionaries of Epochs 12, 13, and 14 into `dinov2_slothead_top3.pth`.

## Consequences

- Resolves the spatial resolution limit for subtle intra-articular injuries while preserving DDP training efficiency.
- Extends training duration to achieve true convergence without GPU out-of-memory errors on Dual Tesla T4 (batch size 16 total, 8 per GPU).
- Establishes a clean, reproducible baseline targeting 0.89–0.90+ Macro AUC for the single ViT architecture before introducing multi-model ensembles.
