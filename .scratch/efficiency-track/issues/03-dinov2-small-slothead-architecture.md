# 03: DINOv2-Small SlotHead Cross-Attention Architecture

**What to build:** The PyTorch neural network architecture (`src/models/dinov2_slothead.py`) wrapping a pretrained DINOv2-Small vision transformer (`vit_small_patch14_dinov2`) at 336×336 px resolution with a 6-slot SlotHead multi-head cross-attention pooling module. Learnable query vectors for each of the 12 clinical findings attend over the 6 slot token representations while respecting sequence presence masks. The model outputs raw logits of shape `(batch_size, 12)`, includes FP16 mixed precision forward capability with automated full-precision FP32 fallback for attention stability, and enables cuDNN benchmarking.

**Blocked by:** 01: End-to-End Wiring Smoke Tracer

**Status:** ready-for-agent

- [ ] Model instantiates `vit_small_patch14_dinov2` with embedding dimension $D=384$ and processes batches of shape `(B, 6, 3, 336, 336)`.
- [ ] SlotHead cross-attention module queries the 6 slot CLS tokens using a learnable parameter matrix $Q \in \mathbb{R}^{12 \times D}$.
- [ ] Attention computation cleanly masks absent slots via sequence presence mask `(B, 6)` such that missing slots receive zero attention weight.
- [ ] Final classifier head returns strictly unscaled raw logits of shape `(B, 12)` (no sigmoid or softmax applied in model forward).
- [ ] Attention weights matrix `(B, 12, 6)` is optionally returnable for diagnostic logging and visualization.
- [ ] Forward pass executes under `torch.autocast("cuda", dtype=torch.float16)` and catches potential attention overflows with an automatic FP32 fallback mechanism.
- [ ] Unit tests in `tests/test_dinov2_slothead.py` verify tensor shapes, gradient flow, and attention mask properties.
