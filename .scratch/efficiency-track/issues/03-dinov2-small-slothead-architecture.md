# 03: DINOv2-Small SlotHead Cross-Attention Architecture

**What to build:** The PyTorch neural network architecture (`src/models/dinov2_slothead.py`) wrapping a DINOv2-Small vision transformer (`vit_small_patch14_dinov2.lvd142m`) at 336×336 px resolution with a 6-slot SlotHead multi-head cross-attention pooling module. Slices for absent sequences are skipped during backbone execution via present-slot gathering to conserve ViT FLOPs. Learnable query vectors for each of the 12 clinical findings attend over the 6 slot token representations with learned slot identity embeddings. The model outputs raw logits of shape `(batch_size, 12)` in FP32 with an unconditional FP32 SlotHead, supports clean diagnostic attention weights via `forward_with_attention`, incorporates empirical label prior bias initialization, supports training-time slot dropout, defaults to `pretrained=False` for offline Kaggle compliance, and includes a DDP all-absent batch gradient synchronization guard.

**Blocked by:** 01: End-to-End Wiring Smoke Tracer, 02: Fast 6-Slot DICOM Preprocessing Pipeline

**Status:** ready-for-execution (approved with 4 execution notes)

- [x] Model instantiates `vit_small_patch14_dinov2.lvd142m` with embedding dimension $D=384$ and processes batches of shape `(B, 6, 3, 336, 336)` with presence mask `(B, 6)`.
- [x] Constructor defaults to `pretrained=False` with support for offline checkpoint loading via `weights_path` and `pretrained_cfg_overlay`.
- [x] Present-slot gathering forwards only active sequence triplets through the ViT backbone (`x[present]`), scattering features back to save compute on absent slots.
- [x] SlotHead cross-attention queries slot tokens using a learnable parameter matrix $Q \in \mathbb{R}^{12 \times D}$ and scaled slot identity embeddings $E_{\text{slot}} \in \mathbb{R}^{6 \times D}$ ($\text{std}=0.20$).
- [x] Attention computation cleanly masks absent slots via sequence presence mask `(B, 6)` such that missing slots receive zero attention weight, with a DDP guard executing a 1-slice dummy forward on all-absent batches.
- [x] Standard `forward(x, presence_mask)` returns strictly unscaled raw logits of shape `(B, 12)` as a single Tensor (FP32, no sigmoid/softmax).
- [x] Diagnostic `forward_with_attention(x, presence_mask)` returns `(logits, attn_weights)` of shape `(B, 12)` and `(B, 12, 6)` with head-averaged weights summing to 1.0 on present slots and strictly 0.0 on all-absent studies.
- [ ] **Criterion 6 (Gated):** Forward pass executes under `torch.autocast("cuda", dtype=torch.float16)` with the SlotHead cross-attention module running in full-precision FP32. Ticked only after passing CUDA T4 verification with calibrated GradScaler (`init_scale=1024.0`) and real pretrained weights probe (`tmp_probe_dinov2_t4.py`).
- [x] Prior-bias initialization initializes `cls_b` to empirical log-odds $\log(p / (1 - p))$ with boundary clamping.
- [x] Slot dropout randomly drops present slots during training (e.g. $p=0.10\text{--}0.15$), guaranteeing at least one present slot remains active per study.
- [x] Unit tests in `tests/test_dinov2_slothead.py` verify shape contracts, gathering efficiency, FP32 stability, GradScaler gradient flow, state dict round-trip inference parity, offline pos_embed resampling, statistical dropout, and prior bias initialization.

### Downstream Ablation Decision Rule (Issues 05 & Beyond)
- Depth coverage $K=3$ (joint-anchored triplets with depth embeddings) or spatial patch attention will be adopted only if mean Macro AUC gain across 3 folds is $\ge +0.010$, strictly positive in every individual fold, and test study latency remains $\le 0.40$s per study on dual T4.
