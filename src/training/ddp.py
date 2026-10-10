"""src/training/ddp.py

DistributedDataParallel stage transition and reduction hook management for RSNA Efficiency Track.
Provides clean architectural separation between src/ modules and training scripts.
"""

from __future__ import annotations

import logging
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


def set_stage_trainable(raw_model: nn.Module, stage: int) -> None:
    """Set parameter requires_grad flags according to unfreezing stage.

    Args:
        raw_model: Unwrapped neural network module containing backbone.
        stage: 1-indexed progressive unfreezing stage (1, 2, 3, or 4).

    Raises:
        ValueError: If stage is not between 1 and 4.
    """
    bb = raw_model.backbone
    for p in bb.parameters():
        p.requires_grad = False

    if stage == 1:
        pass
    elif stage == 2:
        for m in (*bb.blocks[10:], bb.norm):
            for p in m.parameters():
                p.requires_grad = True
    elif stage == 3:
        for m in (*bb.blocks[8:], bb.norm):
            for p in m.parameters():
                p.requires_grad = True
    elif stage == 4:
        for p in bb.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown unfreezing stage: {stage}")


def rewrap_ddp(raw_model: nn.Module, device: torch.device, local_rank: int) -> DDP:
    """Instantiate fresh DDP wrapper with find_unused_parameters=False.

    Args:
        raw_model: Unwrapped neural network module.
        device: Device on which the model resides (CUDA or CPU).
        local_rank: Local device integer index for CUDA ranks.

    Returns:
        Fresh DistributedDataParallel wrapper with newly initialized reduction buckets.
    """
    if device.type == "cuda":
        return DDP(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    return DDP(raw_model, find_unused_parameters=False)


def advance_stage(
    raw_model: nn.Module,
    stage: int,
    local_rank: int,
) -> DDP:
    """Unfreeze parameters for target stage and re-instantiate DDP reduction hooks.

    Caller must set ddp_model = None and run gc.collect() prior to calling advance_stage
    to guarantee old wrapper deallocation.

    Args:
        raw_model: Unwrapped neural network module.
        stage: Target 1-indexed unfreezing stage (1, 2, 3, or 4).
        local_rank: Local device index.

    Returns:
        Newly wrapped DDP model with reduction buckets mapped to currently active parameters.
    """
    set_stage_trainable(raw_model, stage)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if dist.is_initialized():
        dist.barrier()

    new_ddp = rewrap_ddp(raw_model, device, local_rank)

    if stage == 1:
        raw_model.backbone.eval()
    else:
        raw_model.train()

    return new_ddp
