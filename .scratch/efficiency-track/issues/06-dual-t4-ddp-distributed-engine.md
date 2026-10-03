# 06: Dual-T4 DistributedDataParallel (DDP) Engine

**What to build:** A distributed training runner (`scripts/train_ddp_efficiency.py`) wrapping the single-GPU training engine in PyTorch DistributedDataParallel (DDP) across both Kaggle Tesla T4 GPUs via `torchrun` / `torch.multiprocessing.spawn`. 

*Architecture Note on ADR-0003*: ADR-0003 rejected `nn.DataParallel` due to single-process PCIe bus serialization and host RAM exhaustion from scattering/gathering massive 5D MRI tensors on every forward step. In contrast, DDP spawns completely independent OS processes per GPU (`cuda:0` and `cuda:1`); each process loads its own study batches directly into GPU VRAM and only synchronizes full model gradient tensors (~22M parameters, ~90 MB FP32 per sync) via AllReduce across the PCIe bus during backprop, completely eliminating host RAM scatter/gather overhead.

**Blocked by:** 04: TensorBoard Telemetry Suite & Diagnostic Visualizers, 05: Single-GPU Training Pipeline & Top-3 Checkpoint Averaging

**Status:** completed

- [x] Distributed environment initializes 2 worker processes targeting `local_rank` 0 and 1.
- [x] Training studies are sharded across workers using PyTorch `DistributedSampler` with deterministic per-epoch epoch seeding (`sampler.set_epoch(epoch)`).
- [x] Model is wrapped in `torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])`.
- [x] TensorBoard logging, checkpoint saving, and progress bars run strictly on `rank 0` to prevent write collisions.
- [x] Validation evaluation gathers out-of-fold predictions across both workers using `torch.distributed.all_gather_object` before computing fold metrics.
- [x] A dedicated smoke test (`scripts/smoke_ddp.py`) executes 3 epochs across both GPUs at 336 px, confirming >= 1.6x throughput scaling, balanced GPU memory usage, and zero host RAM leaks.
