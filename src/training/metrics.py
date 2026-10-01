"""src/training/metrics.py

Clinical metric calculation engine for RSNA Knee Abnormality Detection.
Handles per-label ROC-AUC, Average Precision, sample distribution statistics,
Gold Study Synovitis filtering, and DDP tensor aggregation.

Single-Responsibility Module: zero dependencies on matplotlib or tensorboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import warnings

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.distributed as dist
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_label_columns(config_path: str | Path | None = None) -> list[str]:
    """Load canonical 12 abnormality column names from config.yaml.

    Args:
        config_path: Path to config.yaml. Defaults to PROJECT_ROOT / 'config.yaml'.

    Returns:
        List of 12 label column names in canonical order.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"

    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cols = cfg.get("labels", {}).get("columns")
    if not cols or len(cols) != 12:
        raise ValueError(f"Expected 12 label columns in {config_path}, found: {cols}")
    return list(cols)


@dataclass(slots=True)
class LabelStats:
    """Per-label sample count and class balance distribution."""

    n_total: int
    n_valid: int
    n_pos: int
    n_neg: int
    pos_ratio: float


@dataclass(slots=True)
class MetricsResult:
    """Structured evaluation results container.

    Attributes:
        macro_auc_12: Macro ROC-AUC across all valid labels (Kaggle competition metric).
        macro_auc_11: Macro ROC-AUC excluding Synovitis (stable for top-3 checkpoint selection).
        per_label_auc: Dict mapping finding name to ROC-AUC score (or NaN).
        per_label_ap: Dict mapping finding name to Average Precision score (or NaN).
        label_stats: Dict mapping finding name to LabelStats distribution.
        n_valid_labels: Number of labels with valid (non-NaN) ROC-AUC scores.
        n_non_finite_logits: Count of NaN or Inf values detected in input logits.
    """

    macro_auc_12: float
    macro_auc_11: float
    per_label_auc: dict[str, float]
    per_label_ap: dict[str, float]
    label_stats: dict[str, LabelStats]
    n_valid_labels: int
    n_non_finite_logits: int


def get_valid_label_mask_and_targets(
    targets: np.ndarray,
    col_idx: int,
    synovitis_idx: int,
    gold_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract valid boolean mask and binarized integer targets for a finding column.

    Enforces:
      1. Acute tear silence: NaN target positions are masked out.
      2. Synovitis evaluation: Restricts evaluation to verified Gold Studies.
      3. Binarization: Ground truth is hard 0/1; continuous pseudo-labels thresholded at >= 0.5.

    Args:
        targets: Array of targets of shape (N, 12).
        col_idx: Index of finding column.
        synovitis_idx: Index of Synovitis in label list.
        gold_mask: Optional boolean array of shape (N,), True for Gold Studies.

    Returns:
        Tuple of:
            valid: Boolean mask of valid rows of shape (N,).
            y_bin: Binarized integer targets of shape (n_valid,).
    """
    valid = ~np.isnan(targets[:, col_idx])
    if col_idx == synovitis_idx and gold_mask is not None:
        valid = valid & gold_mask.astype(bool)

    y_valid = targets[valid, col_idx]
    y_bin = (y_valid >= 0.5).astype(np.int32)
    return valid, y_bin


def compute_competition_metrics(
    logits: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    gold_mask: np.ndarray | torch.Tensor | None = None,
    label_cols: Sequence[str] | None = None,
) -> MetricsResult:
    """Compute per-label and macro metrics adhering strictly to RSNA competition rules.

    Rules enforced:
      1. Acute tear silence: NaN target positions are masked per-label.
      2. Synovitis evaluation: Must be evaluated strictly on Gold Studies (gold_mask=True)
         against binary 0/1 ground truth. Raises ValueError if gold_mask is None.
      3. Rank preservation: ROC-AUC is evaluated directly on raw unscaled logits
         to prevent float32 saturation ties and numerical precision loss.
      4. Single-class safety: Labels with < 2 valid samples or only 1 class yield NaN.
      5. Dual macro signal: Returns macro_auc_11 (checkpoint selection) and macro_auc_12 (competition).

    Args:
        logits: Raw unscaled predictions of shape (N, 12).
        targets: Target labels of shape (N, 12), containing float soft targets or NaNs.
        gold_mask: Boolean array of shape (N,), True for verified Gold Studies.
        label_cols: Sequence of 12 finding names. Defaults to loading from config.yaml.

    Returns:
        MetricsResult container.
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(gold_mask, torch.Tensor):
        gold_mask = gold_mask.detach().cpu().numpy()

    if label_cols is None:
        label_cols = load_label_columns()
    else:
        label_cols = list(label_cols)

    if logits.ndim != 2 or targets.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got logits {logits.shape}, targets {targets.shape}")
    if logits.shape[1] != len(label_cols) or targets.shape[1] != len(label_cols):
        raise ValueError(
            f"Expected {len(label_cols)} columns, got logits {logits.shape[1]}, targets {targets.shape[1]}"
        )

    if gold_mask is not None and gold_mask.shape != (targets.shape[0],):
        raise ValueError(f"Expected gold_mask shape ({targets.shape[0]},), got {gold_mask.shape}")

    if "Synovitis" in label_cols and gold_mask is None:
        raise ValueError(
            "gold_mask is required to evaluate Synovitis on binary ground truth. "
            "Evaluating Synovitis without gold_mask silently evaluates on soft pseudo-labels."
        )

    n_non_finite_logits = int(np.isnan(logits).sum() + np.isinf(logits).sum())
    if n_non_finite_logits > 0:
        warnings.warn(
            f"Detected {n_non_finite_logits} non-finite logits during metric evaluation.",
            stacklevel=2,
        )

    synovitis_idx = label_cols.index("Synovitis") if "Synovitis" in label_cols else -1

    per_label_auc: dict[str, float] = {}
    per_label_ap: dict[str, float] = {}
    label_stats: dict[str, LabelStats] = {}

    # Sigmoid probabilities used strictly for Average Precision
    probs = expit(logits)

    for i, col in enumerate(label_cols):
        valid, y_bin = get_valid_label_mask_and_targets(
            targets=targets,
            col_idx=i,
            synovitis_idx=synovitis_idx,
            gold_mask=gold_mask,
        )

        n_valid = int(valid.sum())
        n_total = len(targets)

        if n_valid == 0:
            per_label_auc[col] = float("nan")
            per_label_ap[col] = float("nan")
            label_stats[col] = LabelStats(n_total, 0, 0, 0, 0.0)
            continue

        n_pos = int((y_bin == 1).sum())
        n_neg = int((y_bin == 0).sum())
        pos_ratio = float(n_pos / n_valid) if n_valid > 0 else 0.0
        label_stats[col] = LabelStats(n_total, n_valid, n_pos, n_neg, pos_ratio)

        # Requires both binary classes to compute meaningful AUC/AP
        if n_pos == 0 or n_neg == 0 or n_valid < 2:
            per_label_auc[col] = float("nan")
            per_label_ap[col] = float("nan")
            continue

        y_score = logits[valid, i]
        p_score = probs[valid, i]

        try:
            per_label_auc[col] = float(roc_auc_score(y_bin, y_score))
        except ValueError:
            per_label_auc[col] = float("nan")

        try:
            per_label_ap[col] = float(average_precision_score(y_bin, p_score))
        except ValueError:
            per_label_ap[col] = float("nan")

    # Macro AUC across all valid labels (competition standard, matching v16)
    valid_aucs_12 = [v for v in per_label_auc.values() if not np.isnan(v)]
    macro_auc_12 = float(np.mean(valid_aucs_12)) if valid_aucs_12 else float("nan")

    # Macro AUC excluding Synovitis (stable for top-3 checkpoint selection)
    valid_aucs_11 = [
        v for k, v in per_label_auc.items() if k != "Synovitis" and not np.isnan(v)
    ]
    macro_auc_11 = float(np.mean(valid_aucs_11)) if valid_aucs_11 else float("nan")

    return MetricsResult(
        macro_auc_12=macro_auc_12,
        macro_auc_11=macro_auc_11,
        per_label_auc=per_label_auc,
        per_label_ap=per_label_ap,
        label_stats=label_stats,
        n_valid_labels=len(valid_aucs_12),
        n_non_finite_logits=n_non_finite_logits,
    )


def all_gather_eval_tensors(
    tensors: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Gather and concatenate evaluation tensors across all DDP processes.

    CRITICAL EXECUTION CONTRACTS:
      1. Collective Call: This function MUST be invoked by all DDP processes concurrently
         before any `if rank == 0` conditional blocks.
      2. Device Placement: Under NCCL, tensors must reside on the active CUDA device.
      3. Type Handling: Boolean tensors are automatically cast to uint8 for transport and restored.
      4. Length Padding: Length padding added for tensor shape uniformity is stripped cleanly.

    Args:
        tensors: Tuple of local rank evaluation tensors, each of shape (N_local, ...).

    Returns:
        Tuple of concatenated tensors of shape (N_total, ...) returned on all ranks.
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensors

    world_size = dist.get_world_size()
    gathered_tensors: list[torch.Tensor] = []

    for t in tensors:
        is_bool = (t.dtype == torch.bool)
        t_transport = t.to(torch.uint8) if is_bool else t
        t_contig = t_transport.contiguous()

        local_size = torch.tensor([t_contig.shape[0]], dtype=torch.long, device=t.device)
        size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(size_list, local_size)

        max_size = max(int(s.item()) for s in size_list)
        padded_shape = list(t_contig.shape)
        padded_shape[0] = max_size
        padded_t = t_contig.new_zeros(padded_shape)
        padded_t[: t_contig.shape[0]] = t_contig

        tensor_list = [torch.zeros_like(padded_t) for _ in range(world_size)]
        dist.all_gather(tensor_list, padded_t)

        unpadded = [
            tensor_list[r][: int(size_list[r].item())] for r in range(world_size)
        ]
        gathered = torch.cat(unpadded, dim=0)
        if is_bool:
            gathered = gathered.bool()
        gathered_tensors.append(gathered)

    return tuple(gathered_tensors)
