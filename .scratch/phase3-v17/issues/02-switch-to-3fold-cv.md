# 02: Standardize 3-Fold Scanner-Fingerprint StratifiedGroupKFold Split

**What to build:**
1. Generate and persist `data/splits/cv_folds_3fold.csv` with 3 balanced folds and zero scanner leakage.
2. Update `config.yaml` to set `cv.n_folds: 3` and `cv.phase3_final_csv: data/splits/cv_folds_3fold.csv`.
3. Update `trainer.py` docstrings and CLI runners to support 3-fold execution.
4. Add assertions verifying all 4,407 studies are present and all folds contain both positive and negative samples for gold labels.

**Blocked by:** 01-metric-repair-and-loss-pos-weights.md

**Status:** ready-for-agent

- [ ] Verify `data/splits/cv_folds_3fold.csv` contains 4,407 rows with ~1,469 studies and ~19–20 gold studies per fold.
- [ ] Update `config.yaml` `cv.n_folds` to `3` and `cv.phase3_final_csv` to `"data/splits/cv_folds_3fold.csv"`.
- [ ] Update docstrings in `src/training/trainer.py` referencing 5-fold to 3-fold.
- [ ] Add unit test validating split properties (no scanner leakage, gold balance across all 3 folds).
- [ ] Ensure all test suite tests pass.
