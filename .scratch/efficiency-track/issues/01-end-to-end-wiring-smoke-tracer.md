# 01: End-to-End Wiring Smoke Tracer

**What to build:** A lightweight, single-command end-to-end verification script (`scripts/smoke_efficiency_tracer.py`) that wires stubbed versions of the 6-slot dataset, DINOv2-Small SlotHead model, masked BCE loss, and TensorBoard logging together. It loads one real or synthetic study on a single GPU, executes one forward pass, computes the loss with soft targets, runs a single backward optimization step, runs a mock inference call, and logs a dummy scalar to TensorBoard. This verifies tensor shapes, presence mask dtypes, device placement, and gradient flow on day one before deep implementation begins.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] Synthetic or single-sample test study successfully constructs a 6-slot image tensor of shape `(1, 6, 3, 336, 336)` and presence mask `(1, 6)`.
- [ ] Model forward pass accepts the 6-slot tensor and presence mask and outputs raw logits of shape `(1, 12)`.
- [ ] Loss function computes valid, non-NaN loss against continuous soft targets with acute tear silence masked out.
- [ ] Single optimizer step completes with non-zero parameter gradient updates without device placement mismatch errors.
- [ ] Mock evaluation produces valid probabilities $[0, 1]$ and writes a smoke test scalar to `outputs/tensorboard/smoke/`.
- [ ] Script runs cleanly via `python3 scripts/smoke_efficiency_tracer.py` and exits with code 0.
