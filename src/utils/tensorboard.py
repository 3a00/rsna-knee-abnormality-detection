"""src/utils/tensorboard.py

TensorBoard telemetry suite and diagnostic visualizers for RSNA Knee Abnormality Detection.
Provides unified TelemetryLogger protocol, NoOp fallback, presence-weighted attention heatmaps,
and PR curves with zero global matplotlib side effects and full exception safety.
"""

from __future__ import annotations

import functools
import logging
import re
from pathlib import Path
from typing import Callable, Protocol, Sequence, runtime_checkable
import warnings

from matplotlib.figure import Figure
import numpy as np
from scipy.special import expit
import torch
import torch.nn as nn
import yaml

from src.training.metrics import (
    MetricsResult,
    get_valid_label_mask_and_targets,
    load_label_columns,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except (ImportError, ModuleNotFoundError):
    SummaryWriter = None
    HAS_TENSORBOARD = False

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def sanitize_tag(tag: str) -> str:
    """Sanitize tag strings to avoid TensorBoard naming rewrite warnings.

    Examples:
        "Baker's" -> "Bakers"
        "Medial Meniscus" -> "Medial_Meniscus"
        "AUC/PF OA" -> "AUC/PF_OA"

    Args:
        tag: Raw tag identifier string.

    Returns:
        Sanitized tag string safe for TensorBoard logging.
    """
    clean = tag.replace("'", "")
    clean = re.sub(r"\s+", "_", clean)
    return clean


def compute_conditional_slot_attention(
    matrix: np.ndarray,
    presence_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute conditional mean slot attention E[attn | present].

    Prevents penalizing rarely present sequences (e.g. COR_T1, SAG_T1)
    with zeros from absent studies.

    Args:
        matrix: Attention weights of shape (B, 12, 6) or (12, 6).
        presence_mask: Optional binary presence mask of shape (B, 6).

    Returns:
        Conditional attention matrix of shape (12, 6).
    """
    if matrix.ndim == 2:
        return matrix
    if matrix.ndim != 3:
        raise ValueError(f"Expected 2D or 3D attention matrix, got shape {matrix.shape}")

    if presence_mask is not None:
        p_expanded = np.expand_dims(presence_mask.astype(np.float32), 1)
        weighted_sum = (matrix * p_expanded).sum(axis=0)  # (12, 6)
        slot_counts = presence_mask.sum(axis=0, keepdims=True).clip(min=1.0)  # (1, 6)
        return weighted_sum / slot_counts

    return matrix.mean(axis=0)


def _safe_telemetry(func: Callable) -> Callable:
    """Decorator ensuring telemetry method failures never crash the training loop."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "writer", None) is None:
            return None
        try:
            return func(self, *args, **kwargs)
        except Exception as err:
            warnings.warn(f"Telemetry call {func.__name__} failed safely: {err}", stacklevel=2)
            return None
    return wrapper


@runtime_checkable
class TelemetryLogger(Protocol):
    """Unified telemetry logger protocol for training monitoring."""

    def log_scalar(self, tag: str, value: float | torch.Tensor, step: int) -> None:
        """Log an individual scalar value."""
        ...

    def log_step(
        self,
        step: int,
        loss: float | torch.Tensor,
        lrs: dict[str, float] | None = None,
        grad_norm: float | None = None,
        grad_scale: float | None = None,
        throughput: float | None = None,
    ) -> None:
        """Log training step loss, parameter group learning rates, and hardware metrics."""
        ...

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: MetricsResult,
        peak_gpu_mem_mb: float | None = None,
        slot_presence_rates: dict[str, float] | None = None,
    ) -> None:
        """Log complete epoch metrics, ROC-AUC, AP, and resource utilization."""
        ...

    def log_graph(self, model: nn.Module, dummy_input: tuple[torch.Tensor, ...]) -> None:
        """Log PyTorch model computational graph."""
        ...

    def log_attention_heatmap(
        self,
        attn_weights: np.ndarray | torch.Tensor,
        step: int,
        presence_mask: np.ndarray | torch.Tensor | None = None,
        tag: str = "Attention/Finding_to_Slot",
        slot_names: Sequence[str] | None = None,
        label_cols: Sequence[str] | None = None,
    ) -> None:
        """Render and log conditional presence-weighted finding-to-slot attention heatmap."""
        ...

    def log_pr_curves(
        self,
        targets: np.ndarray | torch.Tensor,
        logits: np.ndarray | torch.Tensor,
        step: int,
        gold_mask: np.ndarray | torch.Tensor | None = None,
        label_cols: Sequence[str] | None = None,
    ) -> None:
        """Log precision-recall curves for each finding."""
        ...

    def flush(self) -> None:
        """Flush pending events to disk."""
        ...

    def close(self) -> None:
        """Close logger resources."""
        ...


class NoOpLogger:
    """Fallback no-op logger implementing TelemetryLogger protocol with zero overhead."""

    def log_scalar(self, tag: str, value: float | torch.Tensor, step: int) -> None:
        pass

    def log_step(
        self,
        step: int,
        loss: float | torch.Tensor,
        lrs: dict[str, float] | None = None,
        grad_norm: float | None = None,
        grad_scale: float | None = None,
        throughput: float | None = None,
    ) -> None:
        pass

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: MetricsResult,
        peak_gpu_mem_mb: float | None = None,
        slot_presence_rates: dict[str, float] | None = None,
    ) -> None:
        pass

    def log_graph(self, model: nn.Module, dummy_input: tuple[torch.Tensor, ...]) -> None:
        pass

    def log_attention_heatmap(
        self,
        attn_weights: np.ndarray | torch.Tensor,
        step: int,
        presence_mask: np.ndarray | torch.Tensor | None = None,
        tag: str = "Attention/Finding_to_Slot",
        slot_names: Sequence[str] | None = None,
        label_cols: Sequence[str] | None = None,
    ) -> None:
        pass

    def log_pr_curves(
        self,
        targets: np.ndarray | torch.Tensor,
        logits: np.ndarray | torch.Tensor,
        step: int,
        gold_mask: np.ndarray | torch.Tensor | None = None,
        label_cols: Sequence[str] | None = None,
    ) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardLogger(NoOpLogger):
    """Production TensorBoard logger wrapping SummaryWriter with diagnostic visualizers.

    Subclasses NoOpLogger to guarantee protocol compatibility and defaults.
    """

    def __init__(self, log_dir: str | Path) -> None:
        if not HAS_TENSORBOARD or SummaryWriter is None:
            raise RuntimeError(
                "tensorboard package is not installed. Install tensorboard or use NoOpLogger."
            )
        self.log_dir = Path(log_dir)
        self.writer: SummaryWriter | None = None
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        except Exception as err:
            warnings.warn(
                f"Failed to initialize SummaryWriter in {log_dir}: {err}. Telemetry degraded to No-Op.",
                stacklevel=2,
            )

    @_safe_telemetry
    def log_scalar(self, tag: str, value: float | torch.Tensor, step: int) -> None:
        val = float(value.item() if isinstance(value, torch.Tensor) else value)
        if np.isnan(val) or np.isinf(val):
            warnings.warn(
                f"Telemetry received non-finite value ({val}) for tag '{tag}' at step {step}.",
                stacklevel=2,
            )
            return
        clean_tag = sanitize_tag(tag)
        self.writer.add_scalar(clean_tag, val, step)

    @_safe_telemetry
    def log_step(
        self,
        step: int,
        loss: float | torch.Tensor,
        lrs: dict[str, float] | None = None,
        grad_norm: float | None = None,
        grad_scale: float | None = None,
        throughput: float | None = None,
    ) -> None:
        self.log_scalar("Loss/train_step", loss, step)
        if lrs:
            for group_name, lr_val in lrs.items():
                self.log_scalar(f"LR/{group_name}", lr_val, step)
        if grad_norm is not None:
            self.log_scalar("Train/grad_norm", grad_norm, step)
        if grad_scale is not None:
            self.log_scalar("AMP/grad_scale", grad_scale, step)
        if throughput is not None:
            self.log_scalar("Throughput/studies_per_sec", throughput, step)

    @_safe_telemetry
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: MetricsResult,
        peak_gpu_mem_mb: float | None = None,
        slot_presence_rates: dict[str, float] | None = None,
    ) -> None:
        self.log_scalar("Loss/train_epoch", train_loss, epoch)
        self.log_scalar("Loss/val_epoch", val_loss, epoch)

        if not np.isnan(metrics.macro_auc_12):
            self.log_scalar("AUC/macro_val_12", metrics.macro_auc_12, epoch)
        if not np.isnan(metrics.macro_auc_11):
            self.log_scalar("AUC/macro_val_11_checkpoint", metrics.macro_auc_11, epoch)

        self.log_scalar("Metrics/n_valid_labels", float(metrics.n_valid_labels), epoch)

        # Per-label metrics: skip NaNs silently so single-class labels do not trigger false warnings
        for col, auc in metrics.per_label_auc.items():
            if not np.isnan(auc):
                self.log_scalar(f"AUC_Per_Label/{col}", auc, epoch)
        for col, ap in metrics.per_label_ap.items():
            if not np.isnan(ap):
                self.log_scalar(f"AP_Per_Label/{col}", ap, epoch)

        # Log sample distribution once at initial epoch (0 or 1) to prevent redundant flat curves
        if epoch in (0, 1):
            for col, stats in metrics.label_stats.items():
                self.log_scalar(f"Label_Distribution/{col}_pos", float(stats.n_pos), epoch)
                self.log_scalar(f"Label_Distribution/{col}_valid", float(stats.n_valid), epoch)

        if peak_gpu_mem_mb is not None:
            self.log_scalar("System/peak_gpu_mem_mb", peak_gpu_mem_mb, epoch)

        if slot_presence_rates:
            for slot_name, rate in slot_presence_rates.items():
                self.log_scalar(f"Slot_Presence/{slot_name}", rate, epoch)

    @_safe_telemetry
    def log_graph(self, model: nn.Module, dummy_input: tuple[torch.Tensor, ...]) -> None:
        """Log model computational graph on active device with DDP unwrapping."""
        unwrapped_model = getattr(model, "module", model)
        param = next(unwrapped_model.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
        placed_input = tuple(
            inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in dummy_input
        )
        self.writer.add_graph(unwrapped_model, placed_input)
        self.writer.flush()

    @_safe_telemetry
    def log_attention_heatmap(
        self,
        attn_weights: np.ndarray | torch.Tensor,
        step: int,
        presence_mask: np.ndarray | torch.Tensor | None = None,
        tag: str = "Attention/Finding_to_Slot",
        slot_names: Sequence[str] | None = None,
        label_cols: Sequence[str] | None = None,
    ) -> None:
        """Render and log conditional presence-weighted finding-to-slot attention heatmap."""
        if slot_names is None:
            # Lazy import to avoid dragging pydicom/cv2 into lightweight telemetry callers
            from src.datasets.efficiency_pipeline import SLOT_NAMES
            slot_names = SLOT_NAMES
        if label_cols is None:
            label_cols = load_label_columns()

        if isinstance(attn_weights, torch.Tensor):
            matrix = attn_weights.detach().cpu().numpy()
        else:
            matrix = np.array(attn_weights, copy=True)

        if isinstance(presence_mask, torch.Tensor):
            presence_mask = presence_mask.detach().cpu().numpy()

        mean_matrix = compute_conditional_slot_attention(matrix, presence_mask)

        # Pure OO Figure: zero pyplot backend pollution or global leaks
        fig = Figure(figsize=(7.5, 7.0))
        ax = fig.subplots()

        vmax = max(float(mean_matrix.max()), 0.35)
        im = ax.imshow(mean_matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=vmax)
        ax.set_xticks(range(len(slot_names)))
        ax.set_xticklabels(list(slot_names), rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(label_cols)))
        ax.set_yticklabels(list(label_cols), fontsize=9)
        ax.set_title(f"Conditional Slot Attention (Step {step})", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for r in range(mean_matrix.shape[0]):
            for c in range(mean_matrix.shape[1]):
                val = mean_matrix[r, c]
                color = "white" if val > (vmax * 0.6) else "black"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

        fig.tight_layout()
        self.writer.add_figure(sanitize_tag(tag), fig, global_step=step)

    @_safe_telemetry
    def log_pr_curves(
        self,
        targets: np.ndarray | torch.Tensor,
        logits: np.ndarray | torch.Tensor,
        step: int,
        gold_mask: np.ndarray | torch.Tensor | None = None,
        label_cols: Sequence[str] | None = None,
    ) -> None:
        """Log precision-recall curves for each finding using expit probabilities."""
        if label_cols is None:
            label_cols = load_label_columns()
        else:
            label_cols = list(label_cols)

        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        if isinstance(gold_mask, torch.Tensor):
            gold_mask = gold_mask.detach().cpu().numpy()

        synovitis_idx = label_cols.index("Synovitis") if "Synovitis" in label_cols else -1
        if synovitis_idx != -1 and gold_mask is None:
            warnings.warn(
                "gold_mask is None: skipping Synovitis PR curve to avoid evaluating on soft pseudo-labels.",
                stacklevel=2,
            )

        probs = expit(logits)

        for i, name in enumerate(label_cols):
            if i == synovitis_idx and gold_mask is None:
                continue

            valid, y_bin = get_valid_label_mask_and_targets(
                targets=targets,
                col_idx=i,
                synovitis_idx=synovitis_idx,
                gold_mask=gold_mask,
            )

            if len(y_bin) < 2 or len(np.unique(y_bin)) < 2:
                continue

            p = probs[valid, i].astype(np.float32)
            y_tensor = torch.from_numpy(y_bin.astype(np.int64))
            p_tensor = torch.from_numpy(p)

            tag = sanitize_tag(f"PR/{name}")
            self.writer.add_pr_curve(tag, y_tensor, p_tensor, global_step=step)

    @_safe_telemetry
    def flush(self) -> None:
        self.writer.flush()

    @_safe_telemetry
    def close(self) -> None:
        self.writer.close()


def get_logger(
    run_tag: str | None = None,
    base_dir: str | Path | None = None,
    enabled: bool = True,
) -> TelemetryLogger:
    """Factory returning TensorBoardLogger when enabled and available, else NoOpLogger.

    Args:
        run_tag: Unique experiment/fold tag (e.g. 'v17_dinov2_fold0'). If None, returns NoOpLogger.
        base_dir: Output root directory. Defaults to config.yaml outputs.tensorboard_dir or 'outputs/tensorboard'.
        enabled: Boolean flag. Set to (local_rank == 0) to automatically silence DDP worker ranks.

    Returns:
        Configured TelemetryLogger instance (TensorBoardLogger if available and enabled, else NoOpLogger).
    """
    if not enabled or run_tag is None:
        return NoOpLogger()

    if not HAS_TENSORBOARD:
        warnings.warn(
            "TensorBoard is not installed in the environment. Telemetry falling back to NoOpLogger.",
            stacklevel=2,
        )
        return NoOpLogger()

    try:
        if base_dir is None:
            config_path = PROJECT_ROOT / "config.yaml"
            if config_path.is_file():
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                base_dir = cfg.get("outputs", {}).get("tensorboard_dir", "outputs/tensorboard")
            else:
                base_dir = "outputs/tensorboard"

        log_dir = Path(base_dir) / run_tag
        return TensorBoardLogger(log_dir=log_dir)
    except Exception as err:
        warnings.warn(f"Failed to initialize TensorBoardLogger: {err}. Returning NoOpLogger.", stacklevel=2)
        return NoOpLogger()
