# Single-GPU Training on Kaggle Over Multi-GPU DataParallel

We standardized all Kaggle training runs on single-GPU execution (`cuda:0`) rather than multi-GPU `nn.DataParallel`, following catastrophic PCIe bus serialization bottlenecks and host RAM out-of-memory crashes observed during Phase 3 v12.

## Status

accepted

## Context

During the Phase 3 v12 training run on Kaggle's dual T4 setup, wrapping the model in PyTorch `nn.DataParallel` caused batch step times to degrade by 7× (from ~0.70s to 4.88s per batch). PyTorch serialized and scattered 5D MRI tensors (216 slices per study) across the host PCIe bus on every forward pass, exhaustively allocating host system RAM until the Linux kernel terminated the process (`SIGKILL / DeadKernelError`).

## Considered Options

- **`nn.DataParallel` on 2× T4 GPUs**: Caused severe host RAM thrashing and 7× throughput collapse due to MRI tensor scatter/gather overhead.
- **`DistributedDataParallel` (DDP) in Kaggle Notebooks**: High setup friction inside single-notebook execution environments, fragile process spawning.
- **Single-GPU Execution with Gradient Accumulation (Selected)**: Run on GPU 0 with batch size 4 and accumulation steps 2 (effective batch 8). Zero inter-GPU tensor transfer overhead over PCIe.

## Consequences

- Training throughput stabilized at ~0.70s/batch without host RAM leakage.
- Full 5-fold training runs predictably without unexpected kernel crashes.
- The second Kaggle T4 GPU remains idle, which is acceptable within standard Kaggle GPU quota.
