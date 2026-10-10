"""scripts/smoke_ddp.py

Dedicated Dual-GPU DDP Smoke Test for RSNA Knee Abnormality Detection.
Verifies:
  1. Progressive unfreezing gradient flow across stages (assert_grads_flow).
  2. Inter-rank parameter synchronization and parity across stages (assert_rank_parity).
  3. Negative control (--skip-rewrap) rank divergence detection.
  4. Stage 4 batch 8 peak VRAM allocation <= 9.5 GB under FP16 autocast.
  5. Clean bare model checkpoint saving (no 'module.' prefix).
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import gc
import json
import logging
import math
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.datasets.efficiency_pipeline import NUM_SLOTS
from src.models.dinov2_slothead import DINOv2SlotHead
from src.training.ddp import advance_stage, set_stage_trainable
from src.training.losses import MaskedBCEWithLogitsLoss
from src.training.train_efficiency import (
    EfficiencyStudyDataset,
    collate_efficiency,
    seed_everything,
)
from scripts.train_ddp_efficiency import (
    evaluate_dataset_collective,
    lr_for,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"
]


def assert_grads_flow(model: nn.Module, tag: str) -> None:
    """Verify that all trainable parameters received valid finite gradients after backward."""
    dead = [
        n for n, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
    ]
    assert not dead, f"[{tag}] No valid gradient reached: {dead[:5]}"


def assert_rank_parity(model: nn.Module, world_size: int, tag: str) -> None:
    """Verify exact parameter synchronization across worker ranks via a single collective."""
    if world_size <= 1:
        return
    names = [n for n, _ in model.named_parameters()]
    sums = torch.stack([p.detach().double().sum() for p in model.parameters()])
    gathered = [torch.empty_like(sums) for _ in range(world_size)]
    dist.all_gather(gathered, sums)
    for r in range(1, world_size):
        bad = (gathered[r] != gathered[0]).nonzero().flatten().tolist()
        assert not bad, f"[{tag}] Rank {r} diverged from rank 0 on parameters: {[names[i] for i in bad[:5]]}"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def make_synthetic_cache_npz(
    root_dir: Path,
    n_studies: int = 32,
    img_size: int = 336,
) -> tuple[Path, Path, Path]:
    """Create synthetic .npz study archives at 336px for fast smoke testing."""
    cache_dir = root_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    study_uids = [f"synth_{i:04d}" for i in range(n_studies)]
    for uid in study_uids:
        tensor = np.random.randn(6, 5, img_size, img_size).astype(np.float16)
        mask = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        with open(cache_dir / f"{uid}.npz", "wb") as f:
            np.savez_compressed(f, image=tensor, presence_mask=mask, clipped=np.uint8(0))

    manifest = {
        "target_size": img_size,
        "crop_mm": 140.0,
        "n_slices_cached": 5,
        "format": "npz",
        "in_progress": False,
        "failed_uids": [],
        "empty_uids": [],
    }
    with open(cache_dir / "cache_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    splits_csv = root_dir / "splits.csv"
    pd.DataFrame({
        "StudyInstanceUID": study_uids,
        "fold": [0 if i < 8 else 1 for i in range(n_studies)],
        "is_gold": [1 if i % 4 == 0 else 0 for i in range(n_studies)],
    }).to_csv(splits_csv, index=False)

    labels_csv = root_dir / "labels.csv"
    labels_dict = {"StudyInstanceUID": study_uids}
    for col in LABEL_COLS:
        labels_dict[col] = [0.0 if i % 2 == 0 else 1.0 for i in range(n_studies)]
    labels_dict["synovitis_soft"] = [0.63 if i % 2 == 1 else 0.22 for i in range(n_studies)]
    return cache_dir, splits_csv, labels_csv


def make_synthetic_cache(
    root_dir: Path,
    n_studies: int = 64,
    n_val: int = 16,
    img_size: int = 112,
    n_present: int = 4,
) -> tuple[Path, Path, Path]:
    """Create synthetic precomputed study tensors, splits CSV, and labels parquet."""
    cache_dir = root_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    uids = [f"smoke_study_{i:03d}" for i in range(n_studies)]
    for uid in uids:
        tensor = torch.randn(NUM_SLOTS, 5, img_size, img_size, dtype=torch.float16)
        presence = torch.tensor([1.0] * n_present + [0.0] * (NUM_SLOTS - n_present), dtype=torch.float32)
        torch.save({
            "image": tensor,
            "presence_mask": presence,
            "meta": {"target_size": img_size, "crop_mm": 140.0, "n_slices_cached": 5},
        }, cache_dir / f"{uid}.pt")

    manifest = {
        "target_size": img_size,
        "crop_mm": 140.0,
        "n_slices_cached": 5,
        "total_studies_expected": n_studies,
        "total_cached_files": n_studies,
        "failed_uids": [],
        "empty_uids": [],
        "in_progress": False,
        "format": "pt",
    }
    (cache_dir / "cache_manifest.json").write_text(json.dumps(manifest))

    splits_csv = root_dir / "splits.csv"
    fold_ids = [0 if i < n_val else 1 for i in range(n_studies)]
    pd.DataFrame({
        "StudyInstanceUID": uids,
        "fold_id": fold_ids,
        "is_gold": [1 if i < 2 else 0 for i in range(n_studies)],
    }).to_csv(splits_csv, index=False)

    labels_parquet = root_dir / "labels_soft.parquet"
    ldict: dict[str, Any] = {
        "StudyInstanceUID": uids,
        "is_gold": [1 if i < 2 else 0 for i in range(n_studies)],
    }
    for col in LABEL_COLS:
        if col == "Synovitis":
            ldict[col] = [float(i % 2 == 0) if i < 2 else np.nan for i in range(n_studies)]
        else:
            ldict[col] = [float(i % 2 == 0) for i in range(n_studies)]
    ldict["synovitis_soft"] = [0.22] * n_studies
    pd.DataFrame(ldict).to_parquet(labels_parquet)

    return cache_dir, splits_csv, labels_parquet


def _smoke_worker(
    rank: int,
    world_size: int,
    cache_dir: Path,
    splits_csv: Path,
    labels_csv: Path,
    weights_dir: Path,
    results_dict: dict[int, Any],
    img_size: int = 336,
    skip_rewrap: bool = False,
) -> None:
    """Worker function for smoke verification across progressive stages."""
    is_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    backend = "nccl" if is_cuda else "gloo"
    device = torch.device(f"cuda:{rank}") if is_cuda else torch.device("cpu")
    if is_cuda:
        torch.cuda.set_device(rank)

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=30),
    )
    seed_everything(42 + rank)

    raw_model = DINOv2SlotHead(
        pretrained=False,
        img_size=img_size,
        attn_dropout_p=0.2,
        pre_cls_dropout_p=0.3,
    ).to(device)

    # Constant learning rates across all groups ensure parameters diverge if sync fails
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=1e-4)
    criterion = MaskedBCEWithLogitsLoss(gold_weight=2.0)

    # Initialize Stage 1 through advance_stage so all stages follow the identical path
    ddp_model = advance_stage(raw_model, stage=1, local_rank=rank)

    for stage in range(1, 5):
        # 1. Forward & Backward on real step with rank-divergent synthetic data
        torch.manual_seed(1234 + rank + stage * 10)
        synth_images = torch.randn(2, 6, 3, img_size, img_size, device=device)
        synth_masks = torch.ones(2, 6, device=device)
        synth_targets = torch.randint(0, 2, (2, 12), device=device).float()
        synth_gold = torch.tensor([0, 1], device=device)

        optimizer.zero_grad()
        # Autocast disabled on assertion step to prevent underflow to exactly 0.0
        logits = ddp_model(synth_images, synth_masks)
        loss = criterion(logits, synth_targets, is_gold=synth_gold)
        loss.backward()

        # Assert gradient flow on newly unfrozen blocks
        assert_grads_flow(raw_model, tag=f"Stage-{stage}-step-backward")

        optimizer.step()

        # 2. Assert rank parity after optimizer steps, before next stage transition
        assert_rank_parity(raw_model, world_size, tag=f"Stage-{stage}-post-optimizer")

        # 3. Transition to next stage via advance_stage or negative control --skip-rewrap
        if stage < 4:
            next_stage = stage + 1
            if skip_rewrap:
                # Negative control: unfreeze parameters WITHOUT re-wrapping DDP
                set_stage_trainable(raw_model, next_stage)
            else:
                ddp_model = None
                gc.collect()
                if is_cuda:
                    torch.cuda.empty_cache()
                ddp_model = advance_stage(raw_model, next_stage, rank)

    # Acceptance Criterion 10: VRAM Footprint Check under Stage 4 batch 8 with all 6 slots
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
        b8_images = torch.randn(8, 6, 3, img_size, img_size, device=device)
        b8_masks = torch.ones(8, 6, device=device)
        b8_targets = torch.randint(0, 2, (8, 12), device=device).float()
        b8_gold = torch.zeros(8, device=device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            b8_logits = ddp_model(b8_images, b8_masks)
            b8_loss = criterion(b8_logits, b8_targets, is_gold=b8_gold)
        b8_loss.backward()

        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        assert peak_vram_gb <= 9.5, f"Peak VRAM ({peak_vram_gb:.2f} GB) exceeded 9.5 GB threshold!"
        if rank == 0:
            logger.info("VRAM check passed: peak Stage-4 batch-8 memory = %.2f GB <= 9.5 GB", peak_vram_gb)

    # Save bare state dict checkpoint on rank 0
    if rank == 0:
        ckpt_path = weights_dir / "smoke_checkpoint.pth"
        torch.save(raw_model.state_dict(), ckpt_path)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert not any(k.startswith("module.") for k in state.keys()), "Checkpoint leaked 'module.' prefix"

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    results_dict[rank] = {"success": True}


def run_smoke_ddp(img_size: int | None = None, skip_rewrap: bool = False) -> bool:
    """Execute smoke test across 2 processes and assert performance contracts."""
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    world_size = 2 if num_gpus >= 2 or not torch.cuda.is_available() else 1
    actual_img_size = img_size or (336 if num_gpus >= 2 else 56)

    logger.info("=== RSNA DDP Smoke Test Runner ===")
    logger.info("Detected CUDA GPUs: %d (Targeting: %d workers at %dpx, skip_rewrap=%s)",
                num_gpus, world_size, actual_img_size, skip_rewrap)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        cache_dir, splits_csv, labels_csv = make_synthetic_cache_npz(
            tmp_dir, n_studies=32, img_size=actual_img_size
        )
        weights_dir = tmp_dir / "weights"
        weights_dir.mkdir()

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(_find_free_port())

        manager = torch.multiprocessing.get_context("spawn").Manager()
        ddp_results = manager.dict()

        if world_size > 1:
            torch.multiprocessing.spawn(
                _smoke_worker,
                args=(world_size, cache_dir, splits_csv, labels_csv, weights_dir, ddp_results, actual_img_size, skip_rewrap),
                nprocs=world_size,
                join=True,
            )
        else:
            _smoke_worker(
                0, 1, cache_dir, splits_csv, labels_csv, weights_dir, ddp_results, actual_img_size, skip_rewrap
            )

        logger.info("DDP Smoke Verification PASSED cleanly.")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSNA DDP Smoke Test Runner")
    parser.add_argument("--img-size", type=int, default=None, help="Image size (default: 336 on GPU, 56 on CPU)")
    parser.add_argument("--skip-rewrap", action="store_true", default=False, help="Negative control: test desynchronization when DDP rewrap is skipped")
    args = parser.parse_args()
    success = run_smoke_ddp(img_size=args.img_size, skip_rewrap=args.skip_rewrap)
    sys.exit(0 if success else 1)
