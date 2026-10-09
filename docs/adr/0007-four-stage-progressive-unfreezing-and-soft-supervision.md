# 4-Stage Progressive Backbone Unfreezing, Layer-Wise Learning Rate Schedule & Soft Weak Supervision

We refined the DINOv2-Small SlotHead DDP training pipeline across Kaggle Dual Tesla T4 GPUs with an explicit 4-stage progressive backbone unfreezing schedule across a 10-epoch budget, fine-grained 4-tier Layer-Wise Learning Rate Decay (LLRD) with 10 parameter groups, continuous soft weak supervision (`synovitis_soft`), Stage 4-restricted Top-3 checkpoint averaging, and true validation loss telemetry.

## Status

accepted (supersedes ADR-0006)

## Context

In ADR-0006, a 3-stage unfreezing schedule across 8 epochs was accepted to prevent premature ViT representation collapse. However:
1. **Abrupt Unfreezing**: Transitioning from 2 unfrozen blocks directly to all 12 blocks at epoch 5 shocked lower representations and led to suboptimal feature adaptation in intermediate transformer blocks.
2. **Coarse LLRD**: The 3-tier decay grouped blocks into coarse bottom/mid/top buckets without smooth ramps for newly unfrozen layers.
3. **Hard / Silent Synovitis Weak Labels**: Silent Synovitis reports were masked or treated discontinuously, missing calibrated probability distributions (0.63 / 0.22 effusion-conditional probabilities) established in Phase 2 weak label extraction.
4. **Checkpoint Contamination**: Checkpoints could be saved and averaged across mismatched frozen/unfrozen layer regimes (epochs < 7).
5. **Validation Loss Telemetry Gap**: Validation loss was repeating validation macro AUC rather than computing true multi-label BCE loss across gathered validation predictions.

## Considered Options

- **Retain 3-Stage 8-Epoch Schedule**: Faster but suffers representation shocks upon full unfreezing.
- **4-Stage Progressive Unfreezing over 10 Epochs (Selected)**:
  - *Stage 1 (Epochs 1–2)*: Backbone 100% frozen (`eval()` mode). SlotHead queries, cross-attention, and classifier head warmup ($1\times 10^{-3} \to 5\times 10^{-4}$).
  - *Stage 2 (Epochs 3–4)*: Unfreeze top-2 blocks (`blocks[10:]` + `norm`) with a 100-step linear warmup ramp to $1\times 10^{-5}$; head decays to $3\times 10^{-4}$.
  - *Stage 3 (Epochs 5–6)*: Unfreeze top-4 blocks (`blocks[8:]` + `norm`). Newly unfrozen blocks 8–9 (`bb_top1`) ramp to $1\times 10^{-5}$; blocks 10–11 (`bb_top2`) step up to $1.5\times 10^{-5}$; head at flat $2\times 10^{-4}$.
  - *Stage 4 (Epochs 7–10)*: Unfreeze all 12 blocks and embeddings. 4-tier LLRD (Bottom 0–3 at $5\times 10^{-6}$, Mid 4–7 at $1\times 10^{-5}$, Mid-Top 8–9 at $1.5\times 10^{-5}$, Top 10–11 at $2\times 10^{-5}$) with cosine annealing to zero. Newly unfrozen mid/bottom layers follow a 100-step warmup ramp.
- **Continuous Soft Weak Supervision (`synovitis_soft`) (Selected)**:
  - Replace silent non-gold train rows in `Synovitis` with continuous `synovitis_soft` probabilities (0.63 for effusion positive, 0.22 for effusion negative).
  - Strictly preserve ground truth binary targets on gold studies and all validation studies.
  - `--soft-synovitis / --no-soft-synovitis` CLI ablation toggle.
- **Stage 4-Restricted Top-3 Checkpoint Averaging (Selected)**:
  - Candidate checkpoints for Top-3 averaging are saved strictly in Stage 4 (`epoch >= 7`), preventing weight averaging across mismatched parameter regimes.
  - Automatic fallback to single best epoch checkpoint if averaged ensemble does not achieve higher validation score.

## Consequences

- 10-epoch training schedule completes well within the 9-hour Kaggle Dual-T4 budget (~23 minutes total training after one-time caching).
- Smooth layer unfreezing protects pre-trained DINOv2 inductive bias while allowing cross-attention SlotHead queries to stabilize first.
- Continuous soft supervision provides smoother gradient flow on ambiguous findings without corrupting gold validation targets.
- True GPU-safe collective validation loss telemetry enables proper loss convergence monitoring and overfitting detection.
