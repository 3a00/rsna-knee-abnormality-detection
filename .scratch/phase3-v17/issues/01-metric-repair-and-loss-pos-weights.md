# 01: Synovitis Metric Fix, Loss Positive Weights, and GPU Guard

**What to build:**
1. A fail-fast check in notebook setup that verifies the assigned GPU has compute capability $\ge 7.0$ (Turing/Tesla T4 or newer), preventing immediate PyTorch 2.6+ crashes on unsupported Pascal P100 hardware.
2. An update in `src/training/trainer.py` allowing `compute_per_label_auc()` to accept `is_gold_mask`. When evaluating Synovitis (index 8), calculate ROC-AUC exclusively on gold validation rows with binary 0/1 ground truth so that it produces a valid metric instead of `N/A`.
3. Configure square-root scaled positive weights (`labels.pos_weights: [2.70, 6.98, 1.21, 2.34, 2.12, 3.01, 1.83, 1.60, 2.94, 2.49, 2.21, 3.55]`) in `config.yaml` to compensate for class imbalance (especially MCL at 48.8× and Fracture at 12.6×).
4. Wire `pos_weights` from `config.yaml` into `train_fold()` in `src/training/trainer.py`, instantiating `MaskedBCEWithLogitsLoss(pos_weight=pos_weight_tensor)`.
5. Unit tests in `src/datasets/tests/test_phase3.py` verifying both the gold-filtered Synovitis AUC path and that `pos_weight` correctly scales loss without gradient explosion or NaNs.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] Add GPU capability check snippet to notebook setup documentation / template.
- [ ] Update `compute_per_label_auc()` in `src/training/trainer.py` to accept `gold_mask: Optional[np.ndarray] = None` and restrict Synovitis AUC to gold studies.
- [ ] Pass accumulated validation `is_gold` boolean mask to `compute_per_label_auc()` inside `train_fold()`.
- [ ] Add `labels.pos_weights` vector to `config.yaml`.
- [ ] Load `pos_weights` in `train_fold()`, move to active `device`, and pass to `MaskedBCEWithLogitsLoss`.
- [ ] Add unit tests covering gold-masked Synovitis AUC and `pos_weight` loss scaling.
- [ ] All unit tests pass.
