# 07: Dual-Worker High-Speed Test Inference & Kaggle Submission

**What to build:** A self-contained, offline-compatible Kaggle inference notebook and script (`notebooks/efficiency_submission.ipynb` and `scripts/infer_efficiency.py`) that loads pre-trained top-3 averaged DINOv2-Small SlotHead weights, distributes the ~1,300 Kaggle test studies 50/50 across `cuda:0` and `cuda:1` using multiprocessing workers, and writes the formatted `submission.csv` in $\le 15$ minutes without internet access or host memory crashes.

**Blocked by:** 02: Fast 6-Slot DICOM Preprocessing Pipeline, 03: DINOv2-Small SlotHead Cross-Attention Architecture, 06: Dual-T4 DistributedDataParallel (DDP) Engine

**Status:** ready-for-agent

- [ ] Notebook operates completely offline (`HF_HUB_OFFLINE=1`, weights loaded from attached dataset).
- [ ] Test study list is split into two disjoint shards processed concurrently on `cuda:0` and `cuda:1` via `concurrent.futures.ProcessPoolExecutor(max_workers=2)`.
- [ ] Fast 16-worker threaded DICOM reader processes headers with `stop_before_pixels=True` in each worker process.
- [ ] FP16 autocast inference with automatic FP32 retry guarantees zero dropped or failing studies.
- [ ] Outputs strictly adhere to Kaggle submission schema: `StudyInstanceUID` column followed by the 12 finding probabilities $[0, 1]$.
- [ ] Total execution time on 1,300 test studies benchmarks under 15 minutes on dual Tesla T4s.
- [ ] Verification script asserts non-empty, correctly ordered rows with zero NaN, null, or out-of-bounds prediction values.
