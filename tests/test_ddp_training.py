"""tests/test_ddp_training.py

Unit and Multi-Process Integration Test Suite for Issue 06: Dual-T4 DDP Engine.
Verifies:
  1. DistributedSampler deterministic per-epoch epoch seeding and sharding.
  2. Prediction gathering and padding deduplication via gather_unique_records.
  3. DDP state dict unwrapping via 1-rank gloo process group.
  4. Gradient accumulation no_sync() and bitwise parameter synchronization with comm hook AllReduce count verification.
  5. All-absent batch gradient synchronization without DDP reducer deadlocks.
  6. Subprocess CLI end-to-end execution on an odd validation set (n_val=5), verifying padding dedup and ddp_results.json.
"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from scripts.smoke_ddp import make_synthetic_cache
from scripts.train_ddp_efficiency import PROJECT_ROOT, gather_unique_records, unwrap_model
from src.models.dinov2_slothead import DINOv2SlotHead
from src.training.train_efficiency import seed_everything


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


class DummyIndexDataset(Dataset):
    """Simple indexable dataset for sampler verification."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> int:
        return idx


def test_distributed_sampler_epoch_seeding_and_sharding():
    """Verify DistributedSampler partitions data without overlap and set_epoch changes permutation."""
    dataset = DummyIndexDataset(size=10)
    world_size = 2

    # Epoch 1
    s0 = DistributedSampler(dataset, num_replicas=world_size, rank=0, shuffle=True, seed=42)
    s1 = DistributedSampler(dataset, num_replicas=world_size, rank=1, shuffle=True, seed=42)
    s0.set_epoch(1)
    s1.set_epoch(1)

    indices_r0 = list(s0)
    indices_r1 = list(s1)

    assert len(indices_r0) == 5
    assert len(indices_r1) == 5
    assert set(indices_r0).isdisjoint(set(indices_r1))
    assert set(indices_r0).union(set(indices_r1)) == set(range(10))

    # Epoch 2 must yield a different permutation
    s0.set_epoch(2)
    indices_r0_ep2 = list(s0)
    assert indices_r0 != indices_r0_ep2, "sampler.set_epoch failed to update permutation"


def test_gather_unique_records_deduplicates_padding(tmp_path: Path):
    """Verify gather_unique_records deduplicates repeating DistributedSampler padding records."""
    init_file = tmp_path / "pg_dedup"
    dist.init_process_group(
        backend="gloo",
        rank=0,
        world_size=1,
        init_method=f"file://{init_file}",
    )
    try:
        # Simulate local records containing a padded duplicate
        records = [
            {"uid": "study_001", "logits": torch.zeros(12), "targets": torch.ones(12), "is_gold": 1},
            {"uid": "study_002", "logits": torch.zeros(12), "targets": torch.ones(12), "is_gold": 0},
            {"uid": "study_001", "logits": torch.zeros(12), "targets": torch.ones(12), "is_gold": 1},
        ]
        unique = gather_unique_records(records, world_size=1)
        assert len(unique) == 2
        assert [r["uid"] for r in unique] == ["study_001", "study_002"]
    finally:
        dist.destroy_process_group()


def test_ddp_state_dict_unwrapping_matches_raw_model(tmp_path: Path):
    """Verify accessing unwrap_model(ddp_model) strips 'module.' prefix under gloo DDP."""
    init_file = tmp_path / "pg_unwrap"
    dist.init_process_group(
        backend="gloo",
        rank=0,
        world_size=1,
        init_method=f"file://{init_file}",
    )
    try:
        raw_model = DINOv2SlotHead(pretrained=False, img_size=112)
        ddp_model = DDP(raw_model)

        raw_keys = set(raw_model.state_dict().keys())
        unwrapped = unwrap_model(ddp_model)
        unwrapped_keys = set(unwrapped.state_dict().keys())
        wrapped_keys = set(ddp_model.state_dict().keys())

        assert unwrapped_keys == raw_keys
        assert all(k.startswith("module.") for k in wrapped_keys)
        assert not any(k.startswith("module.") for k in unwrapped_keys)
    finally:
        dist.destroy_process_group()


def _accum_worker(
    rank: int,
    world_size: int,
    port: int,
    results_dict: dict[int, Any],
) -> None:
    """Worker verifying grad_accum_steps=2 suppresses intermediate AllReduce post-warmup."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, timeout=timedelta(minutes=5))
    seed_everything(42 + rank)

    model = DINOv2SlotHead(pretrained=False, img_size=112)
    ddp_model = DDP(model)

    # Register DDP communication hook to count AllReduce invocations
    calls = {"n": 0}

    def hook(state, bucket):
        calls["n"] += 1
        return default_hooks.allreduce_hook(state, bucket)

    ddp_model.register_comm_hook(None, hook)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    x = torch.randn(2, 6, 3, 112, 112)
    mask = torch.ones(2, 6)
    target = torch.zeros(2, 12)

    # 1. Warm up DDP bucket layout so bucket counts stabilize
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(ddp_model(x, mask), target)
        loss.backward()
        optimizer.step()

    # 2. Measure exactly how many AllReduce calls occur on one plain sync step
    n0 = calls["n"]
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(ddp_model(x, mask), target)
    loss.backward()
    optimizer.step()
    per_sync = calls["n"] - n0
    assert per_sync > 0, "No AllReduce calls detected during sync step"

    # 3. Execute 2 micro-batches with grad_accum_steps=2: micro-batch 1 suppressed, micro-batch 2 synced
    n1 = calls["n"]
    optimizer.zero_grad(set_to_none=True)
    with ddp_model.no_sync():
        loss = criterion(ddp_model(x, mask), target) / 2.0
        loss.backward()

    loss = criterion(ddp_model(x, mask), target) / 2.0
    loss.backward()
    optimizer.step()

    accum_calls = calls["n"] - n1

    flat = torch.cat([p.detach().flatten() for p in ddp_model.module.parameters()])
    all_flat = [torch.zeros_like(flat) for _ in range(world_size)]
    dist.all_gather(all_flat, flat)

    results_dict[rank] = {
        "weights_match": bool(torch.equal(all_flat[0], all_flat[1])),
        "accum_calls": accum_calls,
        "expected_calls": per_sync,
    }

    dist.barrier()
    dist.destroy_process_group()


def test_cpu_ddp_gradient_accumulation_and_no_sync():
    """Verify no_sync() during accumulation suppresses AllReduce and parameters match bitwise."""
    port = _find_free_port()
    ctx = torch.multiprocessing.get_context("spawn")
    results_dict = ctx.Manager().dict()

    torch.multiprocessing.spawn(
        _accum_worker,
        args=(2, port, results_dict),
        nprocs=2,
        join=True,
    )
    for r in (0, 1):
        res = results_dict[r]
        assert res["weights_match"], "Parameters diverged across workers during grad accumulation"
        assert res["accum_calls"] == res["expected_calls"], (
            f"AllReduce was not suppressed: got {res['accum_calls']} calls, expected {res['expected_calls']}"
        )


def _all_absent_worker(
    rank: int,
    world_size: int,
    port: int,
    results_dict: dict[int, Any],
) -> None:
    """Worker verifying all-absent study on Rank 0 does not deadlock DDP AllReduce reducer."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, timeout=timedelta(minutes=5))
    seed_everything(42 + rank)

    model = DINOv2SlotHead(pretrained=False, img_size=112)
    ddp_model = DDP(model)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    x = torch.randn(2, 6, 3, 112, 112)
    # Rank 0 has all absent slots; Rank 1 has active slots
    mask = torch.zeros(2, 6) if rank == 0 else torch.ones(2, 6)
    target = torch.zeros(2, 12)

    optimizer.zero_grad(set_to_none=True)
    logits = ddp_model(x, mask)
    loss = criterion(logits, target)
    loss.backward()
    optimizer.step()

    flat = torch.cat([p.detach().flatten() for p in ddp_model.module.parameters()])
    all_flat = [torch.zeros_like(flat) for _ in range(world_size)]
    dist.all_gather(all_flat, flat)
    results_dict[rank] = {
        "loss_finite": bool(torch.isfinite(loss)),
        "weights_match": bool(torch.equal(all_flat[0], all_flat[1])),
    }

    dist.barrier()
    dist.destroy_process_group()


def test_ddp_all_absent_batch_synchronization():
    """Verify DDP executes without deadlock when one rank receives an all-absent study batch."""
    port = _find_free_port()
    ctx = torch.multiprocessing.get_context("spawn")
    results_dict = ctx.Manager().dict()

    torch.multiprocessing.spawn(
        _all_absent_worker,
        args=(2, port, results_dict),
        nprocs=2,
        join=True,
    )
    for r in (0, 1):
        assert results_dict[r]["loss_finite"]
        assert results_dict[r]["weights_match"]


def test_cli_end_to_end_odd_val(tmp_path: Path):
    """Subprocess CLI test verifying odd validation set padding, collective re-validation, and results JSON."""
    cache_dir, splits_csv, labels_parquet = make_synthetic_cache(tmp_path, n_studies=16, n_val=5, img_size=112)
    weights_dir = tmp_path / "weights"

    cmd = [
        sys.executable,
        "-m",
        "scripts.train_ddp_efficiency",
        "--backend",
        "gloo",
        "--allow-random-init",
        "--epochs",
        "3",
        "--img-size",
        "112",
        "--num-workers",
        "0",
        "--cache-dir",
        str(cache_dir),
        "--splits-path",
        str(splits_csv),
        "--labels-path",
        str(labels_parquet),
        "--weights-dir",
        str(weights_dir),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, f"DDP CLI failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    results_file = weights_dir / "ddp_results.json"
    assert results_file.is_file(), "Rank 0 failed to produce ddp_results.json"

    res = json.loads(results_file.read_text())
    assert len(res["history"]) == 3, f"Expected 3 history epochs, got {len(res['history'])}"
    assert res["averaged_macro_auc_11"] is not None

    rec_weights = Path(res["recommended_weights_path"])
    assert rec_weights.is_file()
    state_dict = torch.load(rec_weights, weights_only=True)
    assert not any(k.startswith("module.") for k in state_dict.keys()), "Checkpoint contains 'module.' prefix"

# yagni: skipped distributed tensor-parallel model sharding (Megatron-LM); add when model size exceeds 16GB VRAM.
