# 03: Kaggle 3-Fold Training Run and Out-of-Fold Analysis

**What to build:**
1. Update Kaggle training notebook with newly configured 3-fold execution and `pos_weight` loss.
2. Execute full 3-fold training on Kaggle Tesla T4 (single-GPU).
3. Validate training progress: 8 epochs per fold, smooth loss descent, real Synovitis AUC computed every epoch.
4. Perform post-training OOF analysis: compute 3-fold Mean Macro ROC-AUC vs v16 baseline (0.8008 ± 0.0042), fit Platt temperature scaling on gold OOF logits, and export checkpoints (`fold0_best.pth`, `fold1_best.pth`, `fold2_best.pth`, `platt_temperature.pth`).

**Blocked by:** 02-switch-to-3fold-cv.md

**Status:** ready-for-agent

- [ ] Verify Kaggle training notebook incorporates updated `trainer.py`, `config.yaml`, and `cv_folds_3fold.csv`.
- [ ] Run 3-fold training on Kaggle Tesla T4 without timeout (target runtime ~9–10 hours).
- [ ] Verify all 3 folds finish successfully and Synovitis reports a numeric AUC (estimated 0.75–0.85).
- [ ] Confirm Macro ROC-AUC $\ge 0.815$ and MCL AUC $\ge 0.70$.
- [ ] Fit and save `platt_temperature.pth` from OOF gold logits.
- [ ] Save model checkpoints and logs to outputs and Kaggle Dataset.
