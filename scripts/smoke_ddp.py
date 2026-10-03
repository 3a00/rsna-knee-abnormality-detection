"""scripts/smoke_ddp.py

Dedicated Dual-GPU DDP Smoke Test for RSNA Knee Abnormality Detection.
Executes training across 2 worker processes (targeting cuda:0 and cuda:1, or CPU gloo fallback),
verifying:
  1. Process spawning and distributed process group initialization.
  2. Balanced GPU VRAM allocation between workers (~1:1 ratio).
  3. Steady-state host memory stability (zero host RAM leaks, measured via true cumulative RSS post-gc).
  4. >= 1.6x throughput scaling relative to single-worker baseline (measured at 336px with FP16 AMP and full slots).
  5. Exact bitwise parameter synchronization across workers after training.
  6. Rank 0 unwrapped checkpoint generation.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import gc
import os
from pathlib import Path
import socket
import statistics
import sys
import time
from typing import Any

# Ensure project root is first on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure local user site packages can be discovered as fallback if not present (e.g. for psutil)
try:
    import psutil  # type: ignore
except ImportError:
    user_site = Path.home() / f".local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    if user_site.is_dir() and str(user_site) not in sys.path:
        sys.path.append(str(user_site))

import pandas as pd
import psutil
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.datasets.efficiency_pipeline import NUM_SLOTS
from src.models.dinov2_slothead import DINOv2SlotHead
from src.training.losses import MaskedBCEWithLogitsLoss
from src.training.train_efficiency import (
    EfficiencyStudyDataset,
    build_differential_optimizer,
    collate_efficiency,
    seed_everything,
)

LABEL_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"
]


def get_process_rss_mb() -> float:
    """Read cumulative process resident set size (RSS) in MB including child processes."""
    gc.collect()
    p = psutil.Process()
    return sum(q.memory_info().rss for q in [p, *p.children(recursive=True)]) / (1024.0 * 1024.0)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def make_synthetic_cache(
    root_dir: Path,
    n_studies: int = 64,
    n_val: int = 16,
    img_size: int = 112,
    n_present: int = 4,
) -> tuple[Path, Path, Path]:
    """Create synthetic precomputed study tensors, splits CSV, and labels parquet.

    Args:
        root_dir: Temporary directory where cache, splits, and labels are stored.
        n_studies: Total number of synthetic studies to generate.
        n_val: Number of validation studies in the split.
        img_size: Spatial resolution of each slice.
        n_present: Number of present anatomical slots (out of 6).

    Returns:
        Tuple of (cache_dir, splits_csv_path, labels_parquet_path).
    """
    cache_dir = root_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    uids = [f"smoke_study_{i:03d}" for i in range(n_studies)]
    for uid in uids:
        tensor = torch.randn(NUM_SLOTS, 3, img_size, img_size, dtype=torch.float16)
        presence = torch.tensor([1.0] * n_present + [0.0] * (NUM_SLOTS - n_present), dtype=torch.float32)
        torch.save({"image": tensor, "presence_mask": presence, "meta": {"size": img_size}}, cache_dir / f"{uid}.pt")

    splits_csv = root_dir / "splits.csv"
    fold_ids = [0 if i < n_val else 1 for i in range(n_studies)]
    pd.DataFrame({
        "StudyInstanceUID": uids,
        "fold_id": fold_ids,
        "is_gold": [1 if i < 2 else 0 for i in range(n_studies)],
    }).to_csv(splits_csv, index=False)

    labels_parquet = root_dir / "labels_soft.parquet"
    ldict: dict[str, Any] = {"StudyInstanceUID": uids}
    for col in LABEL_COLS:
        ldict[col] = [float(i % 2 == 0) for i in range(n_studies)]
    ldict["synovitis_soft"] = [0.22] * n_studies
    pd.DataFrame(ldict).to_parquet(labels_parquet)

    return cache_dir, splits_csv, labels_parquet


def _smoke_worker(
    rank: int,
    world_size: int,
    cache_dir: Path,
    splits_csv: Path,
    labels_parquet: Path,
    weights_dir: Path,
    results_dict: dict[int, Any],
    img_size: int = 112,
) -> None:
    """Worker function for smoke verification."""
    is_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    backend = "nccl" if is_cuda else "gloo"
    device = torch.device(f"cuda:{rank}") if is_cuda else torch.device("cpu")
    if is_cuda:
        torch.cuda.set_device(rank)

    # Prevent CPU oversubscription
    torch.set_num_threads(2)

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=30),
    )
    seed_everything(42 + rank)

    splits_df = pd.read_csv(splits_csv)
    labels_df = pd.read_parquet(labels_parquet)
    uids = splits_df["StudyInstanceUID"].tolist()
    labels_arr = labels_df[LABEL_COLS].values
    is_gold_arr = splits_df["is_gold"].values
    mock_dirs = [cache_dir / u for u in uids]

    dataset = EfficiencyStudyDataset(
        study_uids=uids,
        study_dirs=mock_dirs,
        labels=labels_arr,
        is_gold=is_gold_arr,
        target_size=img_size,
        cached_tensor_dir=cache_dir,
        num_workers=0,
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
    loader = DataLoader(dataset, batch_size=4, sampler=sampler, collate_fn=collate_efficiency)

    model = DINOv2SlotHead(pretrained=False, img_size=img_size).to(device)
    if is_cuda:
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False,
        )
    else:
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)

    optimizer = build_differential_optimizer(ddp_model.module, lr_backbone=3e-5, lr_head=1e-3, weight_decay=0.02)
    criterion = MaskedBCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=is_cuda)

    epoch_rss: list[float] = []
    step_times: list[float] = []
    step_i = 0
    WARMUP_STEPS = 2

    for epoch in range(1, 4):
        sampler.set_epoch(epoch)
        ddp_model.train()
        for batch in loader:
            t0 = time.perf_counter()
            images = batch["image"].to(device).float()
            masks = batch["presence_mask"].to(device)
            targets = batch["labels"].to(device)
            is_gold = batch["is_gold"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=is_cuda):
                logits = ddp_model(images, masks)
                loss = criterion(logits.float(), targets, is_gold)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if is_cuda:
                torch.cuda.synchronize(device)
            dt = time.perf_counter() - t0

            if step_i >= WARMUP_STEPS:
                step_times.append(dt)
            step_i += 1

        epoch_rss.append(get_process_rss_mb())

    per_rank_tp = 4.0 / max(1e-5, statistics.median(step_times)) if step_times else 0.0
    gpu_vram = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0) if is_cuda else 0.0

    # Bitwise parameter equality check across workers (handles world_size=1 without IndexError)
    flat_params = torch.cat([p.detach().flatten() for p in ddp_model.module.parameters()])
    all_flat = [torch.zeros_like(flat_params) for _ in range(world_size)]
    dist.all_gather(all_flat, flat_params)
    weights_match = all(torch.equal(all_flat[0], t) for t in all_flat[1:])

    # Save checkpoint on rank 0 (strictly bare state dict)
    if rank == 0:
        ckpt_path = weights_dir / "smoke_checkpoint.pth"
        torch.save(ddp_model.module.state_dict(), ckpt_path)

    dist.barrier()
    dist.destroy_process_group()

    results_dict[rank] = {
        "step_times": step_times,
        "per_rank_tp": per_rank_tp,
        "epoch_rss": epoch_rss,
        "gpu_vram_mb": gpu_vram,
        "weights_match": weights_match,
    }


def run_smoke_ddp(img_size: int | None = None) -> bool:
    """Execute smoke test across 2 processes and assert performance contracts.

    Args:
        img_size: Spatial resolution of slices, or None for automatic GPU/CPU selection.

    Returns:
        True if all smoke checks and contracts pass successfully.
    """
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    world_size = 2
    actual_img_size = img_size or (336 if num_gpus >= 2 else 112)
    n_present = NUM_SLOTS if num_gpus >= 2 else 4
    print(f"=== RSNA DDP Smoke Test Runner ===")
    print(f"Detected CUDA GPUs: {num_gpus} (Targeting: {world_size} workers at {actual_img_size}px, {n_present} slots present)")

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        cache_dir, splits_csv, labels_parquet = make_synthetic_cache(
            tmp_dir, n_studies=64, n_val=16, img_size=actual_img_size, n_present=n_present
        )
        weights_dir = tmp_dir / "weights"
        weights_dir.mkdir()

        # 1. Measure 1-worker baseline throughput through the exact same worker function
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(_find_free_port())

        manager = torch.multiprocessing.get_context("spawn").Manager()
        baseline_results = manager.dict()
        torch.multiprocessing.spawn(
            _smoke_worker,
            args=(1, cache_dir, splits_csv, labels_parquet, weights_dir, baseline_results, actual_img_size),
            nprocs=1,
            join=True,
        )
        baseline_tp = baseline_results[0]["per_rank_tp"]
        print(f"1-Worker Baseline Throughput (Median, Post-Warmup): {baseline_tp:.2f} studies/sec")

        # 2. Run 2-worker DDP
        os.environ["MASTER_PORT"] = str(_find_free_port())
        ddp_results = manager.dict()
        torch.multiprocessing.spawn(
            _smoke_worker,
            args=(world_size, cache_dir, splits_csv, labels_parquet, weights_dir, ddp_results, actual_img_size),
            nprocs=world_size,
            join=True,
        )

        r0 = ddp_results[0]
        r1 = ddp_results[1]

        # 3. Assert exact parameter synchronization
        assert r0["weights_match"] and r1["weights_match"], "Model weights diverged across DDP workers"

        # 4. Host RAM leak check: RSS delta between epoch 3 and epoch 2 < 100MB (steady-state)
        rss_leak_0 = r0["epoch_rss"][2] - r0["epoch_rss"][1]
        rss_leak_1 = r1["epoch_rss"][2] - r1["epoch_rss"][1]
        print(f"Host RSS Growth (Ep3 - Ep2): Rank 0 = {rss_leak_0:.2f} MB, Rank 1 = {rss_leak_1:.2f} MB")
        assert rss_leak_0 < 100.0 and rss_leak_1 < 100.0, "Host memory leak detected between steady-state epochs"

        # 5. Throughput scaling check (weak scaling: 2 * per_rank_tp / baseline_tp)
        aggregate_tp = r0["per_rank_tp"] + r1["per_rank_tp"]
        speedup = aggregate_tp / max(1e-5, baseline_tp)
        print(f"2-Worker Aggregate Throughput: {aggregate_tp:.2f} studies/sec (Speedup: {speedup:.2f}x)")
        if num_gpus >= 2:
            assert speedup >= 1.6, (
                f"Throughput scaling fell below 1.6x on dual GPU: {speedup:.2f}x. "
                "(Note: verify --grad-accum-steps 2 is used to amortize PCIe AllReduce overhead)."
            )

        # 6. GPU VRAM balance check
        if num_gpus >= 2:
            v0 = r0["gpu_vram_mb"]
            v1 = r1["gpu_vram_mb"]
            ratio = v0 / max(1e-3, v1)
            print(f"Peak VRAM: GPU 0 = {v0:.1f} MB, GPU 1 = {v1:.1f} MB (Ratio = {ratio:.2f})")
            assert 0.80 <= ratio <= 1.25, f"Unbalanced GPU memory allocation: {v0:.1f} vs {v1:.1f} MB"

        # 7. Verify rank 0 saved checkpoint (strictly bare state dict)
        saved_ckpt = weights_dir / "smoke_checkpoint.pth"
        assert saved_ckpt.is_file(), "Rank 0 failed to produce checkpoint"
        state = torch.load(saved_ckpt, weights_only=True)
        assert not any(k.startswith("module.") for k in state.keys()), "Checkpoint leaked 'module.' prefix"

        print("DDP Smoke Verification PASSED cleanly.")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=None, help="Resolution (336 for GPU benchmark, 112 for CPU test)")
    cli_args = parser.parse_args()
    success = run_smoke_ddp(img_size=cli_args.img_size)
    sys.exit(0 if success else 1)

# yagni: skipped synthetic CUDA PCIe bandwidth saturation benchmark; add when profiling inter-node InfiniBand fabrics.
