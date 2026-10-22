# Multi-Zone Slice Caching, 18-Token SlotHead Cross-Attention & Normalized Task Loss Reweighting

We transition from 5-slice single-triplet joint-center pre-caching to 3-zone compartment-anchored 9-slice pre-caching (25%, 50%, 75% normalized depth), expand DINOv2-Small SlotHead to an 18-token multi-zone representation with additive sequence and depth embeddings, and apply normalized task loss reweighting ($1.75\times$ target boost on Medial Meniscus, Lateral Meniscus, Lateral OA, and PF OA) under a 4-batch $\times$ 2-accumulation DDP training recipe to eliminate physical blindness to peripheral meniscal tears on Kaggle Dual Tesla T4 GPUs.

## Status

accepted (extends ADR-0008, ADR-0009)

## Context

In our 336px Nyquist baseline runs (ADR-0008, ADR-0009), DINOv2-Small reached 0.8465 Macro AUC with central anatomical structures excelling (`ACL Tear`: 0.890, `Joint Effusion`: 0.909), but peripheral structures lagged substantially (`Lateral Meniscus`: 0.739, `Medial Meniscus`: 0.798, `Patellofemoral OA`: 0.806, `Lateral OA`: 0.822).

An anatomical and engineering root-cause audit revealed three fundamental bottlenecks:
1. **Physical Blindness to Peripheral Meniscal Horns**: The existing cache extracted only 5 consecutive physical slices centered strictly on the joint normal ($[c-2 \dots c+2]$). While cruciate ligaments and fluid reside in the intercondylar notch ($c \pm 2$), anterior and posterior meniscal horns, roots, and lateral articular margins lie in the periphery ($c \pm 6$ to $c \pm 12$). These outer slices were physically absent from the cache, making the model blind to peripheral tears.
2. **The Single-Triplet Inference Dilemma**: If peripheral slices are cached, the model cannot simply evaluate a single 3-slice window during validation, because a central triplet cannot resolve meniscal tears, while a peripheral triplet cannot resolve ACL tears or notch effusion. The network requires simultaneous representation of central and peripheral compartments in a single forward pass.
3. **Uniform Task Loss Dilution**: Under standard multi-label BCE, easily learned findings (Effusion, ACL) dominate gradient updates, while noisy, complex intra-articular injuries (Menisci, OA) receive diluted gradients, slowing convergence on boundary-adjacent tears.
4. **Tesla T4 16 GB VRAM Ceiling**: Tripling the token footprint from 6 slots to 18 slot-zone tokens triples ViT patch forward computations. Retaining batch size 8 per GPU risks out-of-memory (OOM) failures on 16 GB Tesla T4 hardware.

## Considered Options

- **Continuous 9-Slice Slicing ($[c-4 \dots c+4]$)**: Caches 9 contiguous slices around the center. Rejected: in a 30-slice series, $[c-4 \dots c+4]$ only spans $37\%\text{--}63\%$ of depth, still omitting peripheral meniscal horns and collateral margins.
- **3-Zone Compartment-Anchored Sampling (Selected)**:
  - Anchor 3 discrete triplets at normalized depth percentiles along the physical normal projection vector ($k$):
    - Zone 0 (Lateral / Anterior): $c_0 = \text{round}(0.25 \times (N - 1))$
    - Zone 1 (Center / Notch): $c_1 = \text{round}(0.50 \times (N - 1))$
    - Zone 2 (Medial / Posterior): $c_2 = \text{round}(0.75 \times (N - 1))$
  - Extract 3 consecutive slices $[c_z-1, c_z, c_z+1]$ per zone, yielding exactly 9 slices per slot ($6 \times 9 = 54$ slices total per study). Edge-clamped via `np.clip(idx, 0, N - 1)` for short series.
- **Multi-Zone Feature Pooling vs Multi-Pass Inference**:
  - *Pooling*: Average-pool or max-pool 3 zone tokens per slot into 1 token. Rejected: washes out focal 1–3 mm tears across large background volumes.
  - *Multi-Pass*: Run 3 forward passes at inference (Zone 0, Zone 1, Zone 2). Rejected: triples test-time inference latency from 14 minutes to 42 minutes, violating Efficiency Track constraints.
- **18-Token Additive Multi-Zone SlotHead (Selected)**:
  - Forward all 18 triplets through the shared DINOv2-Small backbone (using non-reentrant gradient checkpointing).
  - Add factored learnable embeddings: $\mathbf{t}_{s, z} = \mathbf{f}_{s, z} + \mathbf{e}_{\text{slot}, s} + \mathbf{e}_{\text{zone}, z}$, with $\mathbf{e}_{\text{slot}} \in \mathbb{R}^{6 \times D}$ and $\mathbf{e}_{\text{zone}} \in \mathbb{R}^{3 \times D}$.
  - SlotHead cross-attention queries $\mathbf{Q} \in \mathbb{R}^{12 \times D}$ attend across all 18 keys/values, enabling finding queries (e.g. Medial Meniscus) to attend specifically to corresponding peripheral tokens.
  - Absent sequences mask all 3 corresponding zone tokens simultaneously in `key_padding_mask`. Slot dropout drops entire 3-zone slot clusters.
- **Normalized Task-Level Loss Reweighting (Selected)**:
  - Add normalized task multipliers $\mathbf{w}_{\text{task}} \in \mathbb{R}^{12}$ to `MaskedBCEWithLogitsLoss`.
  - Upweight lowest 4 performers (`Medial Meniscus`, `Lateral Meniscus`, `Lateral OA`, `PF OA`) with a $1.75\times$ target multiplier, keeping $1.0\times$ for the other 8 labels.
  - Normalize so $\frac{1}{12}\sum_{i=1}^{12} w_i = 1.0$ (giving $1.40\times$ for target 4, $0.80\times$ for remaining 8), preserving optimizer learning rate dynamics and gradient magnitude.
- **Dedicated 9-Slice Scratch Cache (`/kaggle/tmp/rsna_cache_336_9slice`) (Selected)**:
  - Persist $(6, 9, 336, 336)$ float16 arrays with manifest validation (`n_slices_cached: 9`, `sampling_strategy: "3zone"`), preventing silent corruption against legacy 5-slice caches.
- **DDP Batch Configuration ($4 \times 2$) (Selected)**:
  - Set `--batch-size 4` with `--accum-steps 2` on Dual T4 GPUs. Maintains identical effective batch size 16 across 2 GPUs while capping peak VRAM at ~9.5 GB per GPU (safely below 16 GB).

- **Config-Driven Gold Weight Supervision (Selected)**:
  - Preserve `MaskedBCEWithLogitsLoss(gold_weight=2.0)` as library class default for backward compatibility with existing callers.
  - Set `gold_weight = 5.0` in `config.yaml` (`training.gold_weight: 5.0`) and the training notebook CLI (`--gold-weight 5.0`) to provide strong, unambiguous supervision on the $n=58$ expert-annotated gold studies during 14-epoch training, dynamically resolved via `resolve_gold_weight(args)`.

## Consequences

- Completely eliminates physical blindness to peripheral meniscal horns and outer articular cartilage margins.
- Enables finding-specific spatial attention across anatomical depth zones in a single forward pass without test-time serialization penalties.
- Preserves full backward mathematical parity for loss scales via normalized task multipliers.
- Total scratch cache size is ~30.8 GB for 4,407 studies, fitting comfortably within Kaggle's 73 GB `/kaggle/tmp` scratch limit.
- One-time pre-caching duration increases moderately to ~135 minutes (well within the 9-hour session limit), while training epoch duration remains ~4.5 minutes.
