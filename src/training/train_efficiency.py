"""src/training/train_efficiency.py

Single-GPU Training Engine for RSNA Knee Abnormality Detection (Efficiency Track).
Trains DINOv2-Small SlotHead on 6-slot MRI sequences with calibrated weak supervision,
differential AdamW optimizer, OneCycleLR schedule, and Top-3 checkpoint averaging.

Single-Responsibility Module: Orchestrates training and validation steps.
Metric calculations, loss definitions, and visualizers are imported from dedicated modules.

Notes on Gradient Accumulation:
    A trailing partial accumulation group is scaled by 1/grad_accum_steps,
    so its gradient is slightly smaller; this affects at most one optimizer step per epoch.
"""

from __future__ import annotations

import argparse
import functools
import logging
import math
from pathlib import Path
import random
from typing import Any, Sequence
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

from src.datasets.efficiency_pipeline import (
    DEFAULT_CROP_MM,
    SLOT_NAMES,
    Fast6SlotDICOMDataset,
)
from src.models.dinov2_slothead import DINOv2SlotHead
from src.training.losses import MaskedBCEWithLogitsLoss
from src.training.metrics import (
    MetricsResult,
    compute_competition_metrics,
    load_label_columns,
)
from src.utils.tensorboard import NoOpLogger, TelemetryLogger, TensorBoardLogger

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def seed_everything(seed: int) -> None:
    """Set random seeds across Python, NumPy, PyTorch CPU and CUDA for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int, base_seed: int) -> None:
    """Spawn-safe DataLoader worker seed initialization."""
    s = base_seed + worker_id
    np.random.seed(s)
    random.seed(s)


def load_pos_weights(config_path: str | Path | None = None) -> list[float]:
    """Read square-root damped positive class weights from config.yaml (single source of truth)."""
    cfg_p = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    with open(cfg_p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    weights = cfg.get("labels", {}).get("pos_weights")
    if not weights or len(weights) != 12:
        raise ValueError(f"Expected 12 pos_weights in {cfg_p}, found: {weights}")
    return [float(x) for x in weights]


class EfficiencyStudyDataset(Dataset):
    """6-Slot dataset wrapping fast selective DICOM decoding with is_gold indicator.

    Supports pre-extracted tensor caching: if cached_tensor_dir contains {uid}.pt,
    loads the precomputed (image, presence_mask) tuple directly, bypassing DICOM I/O.
    Validates cache provenance (target_size and crop_mm) if meta dictionary is present.
    The underlying Fast6SlotDICOMDataset is constructed lazily on the first cache miss.
    """

    def __init__(
        self,
        study_uids: Sequence[str],
        study_dirs: Sequence[str | Path],
        labels: np.ndarray | None = None,
        is_gold: np.ndarray | None = None,
        series_csv_path: str | Path | None = None,
        target_size: int = 336,
        crop_mm: float = DEFAULT_CROP_MM,
        cached_tensor_dir: str | Path | None = None,
        num_workers: int = 0,
    ) -> None:
        self.study_uids = [str(u) for u in study_uids]
        self.study_dirs = [Path(d) for d in study_dirs]
        self.labels = np.asarray(labels, dtype=np.float32) if labels is not None else None
        self.is_gold = np.asarray(is_gold, dtype=np.int64) if is_gold is not None else None
        self.series_csv_path = series_csv_path
        self.target_size = target_size
        self.crop_mm = crop_mm
        self.cached_dir = Path(cached_tensor_dir) if cached_tensor_dir else None
        self.num_workers = num_workers
        self._dicom_ds: Fast6SlotDICOMDataset | None = None

    def _get_dicom_ds(self) -> Fast6SlotDICOMDataset:
        """Lazily instantiate DICOM loader with num_workers=0 to prevent thread pool nesting."""
        if self._dicom_ds is None:
            self._dicom_ds = Fast6SlotDICOMDataset(
                study_dirs=self.study_dirs,
                labels=self.labels,
                series_csv_path=self.series_csv_path,
                num_workers=0,  # Avoid thread-pool oversubscription when DataLoader workers > 0
                target_size=self.target_size,
                crop_mm=self.crop_mm,
            )
        return self._dicom_ds

    def __len__(self) -> int:
        return len(self.study_uids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        uid = self.study_uids[idx]

        # Fast-path: pre-extracted cached tensor
        if self.cached_dir is not None:
            cache_file = self.cached_dir / f"{uid}.pt"
            if cache_file.is_file():
                cached = torch.load(cache_file, map_location="cpu", weights_only=True)
                if isinstance(cached, dict):
                    meta = cached.get("meta")
                    if meta is not None and (
                        int(meta.get("size", self.target_size)) != self.target_size
                        or abs(float(meta.get("crop_mm", self.crop_mm)) - self.crop_mm) > 1e-6
                    ):
                        raise RuntimeError(
                            f"Stale cache for {uid}: cached {meta}, requested size={self.target_size}, crop_mm={self.crop_mm}"
                        )
                    img = cached["image"]
                    mask = cached["presence_mask"]
                else:
                    img, mask = cached[0], cached[1]

                item: dict[str, Any] = {
                    "study_id": uid,
                    "image": img,  # float16 kept through loader, cast in train/val loops
                    "presence_mask": mask.float(),
                }
                if self.labels is not None:
                    item["labels"] = torch.from_numpy(self.labels[idx])
                if self.is_gold is not None:
                    item["is_gold"] = torch.tensor(self.is_gold[idx], dtype=torch.long)
                else:
                    item["is_gold"] = torch.tensor(0, dtype=torch.long)
                return item

        # Standard DICOM pipeline fallback (lazy instantiation)
        base_item = self._get_dicom_ds()[idx]
        item = {
            "study_id": uid,
            "image": base_item["image"],
            "presence_mask": base_item["presence_mask"].float(),
        }
        if self.labels is not None:
            item["labels"] = torch.from_numpy(self.labels[idx])
        if self.is_gold is not None:
            item["is_gold"] = torch.tensor(self.is_gold[idx], dtype=torch.long)
        else:
            item["is_gold"] = torch.tensor(0, dtype=torch.long)
        return item


def collate_efficiency(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate batch items into stacked PyTorch tensors."""
    collated: dict[str, Any] = {
        "study_id": [b["study_id"] for b in batch],
        "image": torch.stack([b["image"] for b in batch]),
        "presence_mask": torch.stack([b["presence_mask"] for b in batch]),
    }
    if "labels" in batch[0] and batch[0]["labels"] is not None:
        collated["labels"] = torch.stack([b["labels"] for b in batch])
    if "is_gold" in batch[0] and batch[0]["is_gold"] is not None:
        collated["is_gold"] = torch.stack([b["is_gold"] for b in batch])
    return collated


def load_labels_and_splits(
    labels_path: str | Path,
    splits_path: str | Path,
    fold_id: int = 0,
    label_cols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load and merge calibrated soft labels with StratifiedGroupKFold splits.

    synovitis_soft is authoritative for non-gold rows; gold rows are never modified.

    Args:
        labels_path: Path to labels_soft.parquet or pseudo_labels.csv.
        splits_path: Path to cv_folds_3fold.csv.
        fold_id: Target validation fold (0, 1, or 2).
        label_cols: Sequence of 12 target finding names. Defaults to config.yaml.

    Returns:
        Tuple of (train_df, val_df, label_cols).
    """
    if label_cols is None:
        label_cols = load_label_columns()
    else:
        label_cols = list(label_cols)

    labels_p = Path(labels_path)
    if not labels_p.is_file():
        raise FileNotFoundError(f"Labels file not found: {labels_p}")

    if labels_p.suffix == ".parquet":
        labels_df = pd.read_parquet(labels_p)
    else:
        labels_df = pd.read_csv(labels_p)

    splits_p = Path(splits_path)
    if not splits_p.is_file():
        raise FileNotFoundError(f"Splits file not found: {splits_p}")
    splits_df = pd.read_csv(splits_p)

    # Drop pre-existing split columns to prevent _x/_y suffix collision
    labels_df = labels_df.drop(columns=[c for c in ("fold_id", "is_gold") if c in labels_df.columns])

    # Merge labels with split fold assignment and gold status
    req_split_cols = ["StudyInstanceUID", "fold_id", "is_gold"]
    merged = labels_df.merge(splits_df[req_split_cols], on="StudyInstanceUID", how="inner", validate="one_to_one")
    if len(merged) != len(labels_df):
        logger.warning("%d label rows had no split assignment and were excluded", len(labels_df) - len(merged))

    # Impute synovitis_soft for non-gold studies where synovitis_soft is available
    if "synovitis_soft" in merged.columns and "Synovitis" in label_cols:
        non_gold_mask = ~merged["is_gold"].astype(bool)
        has_soft = merged["synovitis_soft"].notna()
        changed = (
            merged.loc[non_gold_mask & has_soft, "Synovitis"]
            != merged.loc[non_gold_mask & has_soft, "synovitis_soft"]
        ).sum()
        logger.info(
            "Synovitis assigned from synovitis_soft on %d non-gold rows (%d silent studies updated)",
            int((non_gold_mask & has_soft).sum()),
            int(changed),
        )
        merged.loc[non_gold_mask & has_soft, "Synovitis"] = merged.loc[non_gold_mask & has_soft, "synovitis_soft"]

    train_df = merged[merged["fold_id"] != fold_id].copy().reset_index(drop=True)
    val_df = merged[merged["fold_id"] == fold_id].copy().reset_index(drop=True)

    logger.info(
        "Fold %d loaded: %d train studies (%d gold), %d val studies (%d gold)",
        fold_id,
        len(train_df),
        int(train_df["is_gold"].sum()),
        len(val_df),
        int(val_df["is_gold"].sum()),
    )
    return train_df, val_df, label_cols


def build_differential_optimizer(
    model: nn.Module,
    lr_backbone: float = 3e-5,
    lr_head: float = 1e-3,
    weight_decay: float = 0.02,
) -> torch.optim.AdamW:
    """Build AdamW optimizer with layer-wise differential learning rates and weight decay exclusions.

    Groups:
      0: Backbone 2D weights (lr=lr_backbone, weight_decay=weight_decay)
      1: Backbone 1D/biases/LayerNorm/pos_embed (lr=lr_backbone, weight_decay=0.0)
      2: SlotHead 2D weights (lr=lr_head, weight_decay=weight_decay)
      3: SlotHead 1D/biases/LayerNorm/queries/slot_embed (lr=lr_head, weight_decay=0.0)

    Note for Issue 06:
      This optimizer must be constructed BEFORE wrapping the model in DistributedDataParallel,
      as parameter names will be prefixed with 'module.backbone.'.
    """
    no_decay_head = {"slot_embed", "finding_queries"}

    bb_decay, bb_no_decay = [], []
    hd_decay, hd_no_decay = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = name.startswith("backbone.")
        is_no_decay = (
            p.ndim < 2
            or "bias" in name
            or "norm" in name
            or "pos_embed" in name
            or "cls_token" in name
            or any(t in name for t in no_decay_head)
        )
        if is_backbone:
            (bb_no_decay if is_no_decay else bb_decay).append(p)
        else:
            (hd_no_decay if is_no_decay else hd_decay).append(p)

    param_groups = [
        {"params": bb_decay, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": bb_no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        {"params": hd_decay, "lr": lr_head, "weight_decay": weight_decay},
        {"params": hd_no_decay, "lr": lr_head, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups)


def _register_checkpoint(
    top: list[tuple[float, int, Path]],
    score: float,
    epoch: int,
    path: Path,
    k: int = 3,
) -> None:
    """Insert checkpoint into top-k tracking list and delete whichever checkpoint falls out."""
    top.append((score, epoch, path))
    top.sort(key=lambda x: x[0], reverse=True)
    while len(top) > k:
        _, _, evicted = top.pop()
        evicted.unlink(missing_ok=True)


def average_top_checkpoints(
    checkpoint_paths: Sequence[str | Path],
    output_path: str | Path,
) -> Path:
    """Average floating-point parameter weights across top-k checkpoint state dicts."""
    if not checkpoint_paths:
        raise ValueError("Cannot average empty list of checkpoint paths.")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    state_dicts = [
        torch.load(p, map_location="cpu", weights_only=True)["model_state_dict"]
        for p in checkpoint_paths
    ]
    ref_dict = state_dicts[0]
    n_ckpts = float(len(state_dicts))

    averaged_state: dict[str, torch.Tensor] = {}
    for k, v in ref_dict.items():
        if torch.is_floating_point(v):
            averaged_state[k] = (sum(d[k].float() for d in state_dicts) / n_ckpts).to(v.dtype)
        else:
            averaged_state[k] = v.clone()

    torch.save(averaged_state, out_p)
    logger.info("Averaged %d checkpoints into %s", len(checkpoint_paths), out_p)
    return out_p


@torch.no_grad()
def run_validation(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    dev: torch.device,
    use_amp: bool,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run validation over a DataLoader and accumulate predictions and losses."""
    model.eval()
    val_loss_total = 0.0
    val_samples = 0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_is_gold: list[torch.Tensor] = []

    for batch in loader:
        images = batch["image"].to(dev, non_blocking=True).float()
        masks = batch["presence_mask"].to(dev, non_blocking=True)
        targets = batch["labels"].to(dev, non_blocking=True)
        is_gold = batch["is_gold"].to(dev, non_blocking=True)

        with torch.amp.autocast(device_type=dev.type, dtype=torch.float16, enabled=(use_amp and dev.type == "cuda")):
            logits = model(images, masks)
            loss = criterion(logits.float(), targets, is_gold)

        val_loss_total += loss.item() * len(targets)
        val_samples += len(targets)

        all_logits.append(logits.float().cpu())
        all_targets.append(targets.float().cpu())
        all_is_gold.append(is_gold.cpu())

    epoch_val_loss = val_loss_total / max(1, val_samples)
    concat_logits = torch.cat(all_logits, dim=0) if all_logits else torch.empty(0, 12)
    concat_targets = torch.cat(all_targets, dim=0) if all_targets else torch.empty(0, 12)
    concat_is_gold = torch.cat(all_is_gold, dim=0) if all_is_gold else torch.empty(0, dtype=torch.long)
    return epoch_val_loss, concat_logits, concat_targets, concat_is_gold


def train_efficiency_fold(
    fold_id: int = 0,
    n_epochs: int = 12,
    batch_size: int = 4,
    grad_accum_steps: int = 1,
    lr_backbone: float = 3e-5,
    lr_head: float = 1e-3,
    weight_decay: float = 0.02,
    pct_start: float = 0.15,
    gold_weight: float = 5.0,
    grad_clip_norm: float = 1.0,
    img_size: int = 336,
    weights_path: str | Path | None = None,
    allow_random_init: bool = False,
    num_workers: int = 2,
    config_path: str | Path | None = None,
    limit_train: int | None = None,
    limit_val: int | None = None,
    labels_path: str | Path | None = None,
    splits_path: str | Path | None = None,
    dicom_root: str | Path | None = None,
    series_csv: str | Path | None = None,
    cached_tensor_dir: str | Path | None = None,
    weights_dir: str | Path | None = None,
    tb_log_dir: str | Path | None = None,
    device: str | torch.device = "cuda:0",
    use_amp: bool = True,
    seed: int = 42,
    study_dirs_map: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Train DINOv2-Small SlotHead model on a single GPU for one cross-validation fold.

    Args:
        fold_id: Target cross-validation fold (0, 1, or 2).
        n_epochs: Total training epochs (default: 12).
        batch_size: Studies per batch (default: 4).
        grad_accum_steps: Number of forward batches before optimizer step (default: 1).
        lr_backbone: ViT backbone peak learning rate (default: 3e-5).
        lr_head: SlotHead cross-attention peak learning rate (default: 1e-3).
        weight_decay: Weight decay for 2D weights (default: 0.02).
        pct_start: OneCycleLR warmup percentage (default: 0.15).
        gold_weight: Loss multiplier for 58 verified gold studies (default: 5.0).
        grad_clip_norm: Maximum gradient norm clipping (default: 1.0).
        img_size: Image slice spatial resolution (default: 336).
        weights_path: Path to offline DINOv2 backbone checkpoint.
        allow_random_init: Permit training without backbone weights (strictly for tests).
        num_workers: DataLoader background worker processes (default: 2).
        config_path: Path to config.yaml (default: PROJECT_ROOT / config.yaml).
        limit_train: Optional subset size for training dry runs.
        limit_val: Optional subset size for validation dry runs.
        labels_path: Path to soft labels file.
        splits_path: Path to CV splits CSV.
        dicom_root: Root directory of DICOM files.
        series_csv: Path to train_series.csv metadata.
        cached_tensor_dir: Directory containing pre-extracted {uid}.pt tensors.
        weights_dir: Output directory for saving model checkpoints.
        tb_log_dir: TensorBoard log directory.
        device: Active torch device (e.g. 'cuda:0' or 'cpu').
        use_amp: Enable Automatic Mixed Precision (FP16).
        seed: Random seed for reproducibility.
        study_dirs_map: Optional pre-mapped dictionary of study UID to DICOM directory.

    Returns:
        Dictionary containing best metrics, top-3 checkpoint paths, history, and recommended weight path.
    """
    if weights_path is None and not allow_random_init:
        raise ValueError(
            "weights_path (offline DINOv2 backbone checkpoint) is required. "
            "Pass allow_random_init=True only for unit/smoke tests."
        )
    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")
    if weights_path is not None and not Path(weights_path).is_file():
        raise FileNotFoundError(f"Backbone weights not found: {weights_path}")

    seed_everything(seed)
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True

    ckpt_dir = (
        Path(weights_dir)
        if weights_dir
        else PROJECT_ROOT / "outputs" / "weights" / f"dinov2_slothead_fold{fold_id}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Telemetry logger setup
    logger_instance: TelemetryLogger
    if tb_log_dir is not None:
        try:
            logger_instance = TensorBoardLogger(tb_log_dir)
        except Exception as err:
            warnings.warn(f"TensorBoard unavailable ({err}); falling back to NoOpLogger.", stacklevel=2)
            logger_instance = NoOpLogger()
    else:
        logger_instance = NoOpLogger()

    # Load data splits and labels
    train_df, val_df, label_cols = load_labels_and_splits(
        labels_path=labels_path or (PROJECT_ROOT / "data/labels/labels_soft.parquet" if (PROJECT_ROOT / "data/labels/labels_soft.parquet").is_file() else PROJECT_ROOT / "data/labels/pseudo_labels.csv"),
        splits_path=splits_path or (PROJECT_ROOT / "data/splits/cv_folds_3fold.csv"),
        fold_id=fold_id,
    )

    # Dry-run subsetting preserving gold studies
    def _subset(df: pd.DataFrame, n: int | None, s: int) -> pd.DataFrame:
        if n is None or n >= len(df):
            return df
        gold = df[df["is_gold"] == 1]
        rest = df[df["is_gold"] != 1].sample(n=max(0, n - len(gold)), random_state=s)
        return pd.concat([gold, rest]).sample(frac=1.0, random_state=s).head(n).reset_index(drop=True)

    train_df = _subset(train_df, limit_train, seed)
    val_df = _subset(val_df, limit_val, seed)

    # Resolve study directories for train and validation
    def resolve_study_dirs(df: pd.DataFrame) -> list[Path]:
        dirs = []
        for uid in df["StudyInstanceUID"]:
            if study_dirs_map and str(uid) in study_dirs_map:
                dirs.append(study_dirs_map[str(uid)])
            else:
                primary = Path(dicom_root or PROJECT_ROOT / "data/raw") / "train_series" / str(uid)
                fallback = Path(dicom_root or PROJECT_ROOT / "data/raw") / str(uid)
                dirs.append(primary if primary.is_dir() else fallback)
        return dirs

    train_dirs = resolve_study_dirs(train_df)
    val_dirs = resolve_study_dirs(val_df)

    train_ds = EfficiencyStudyDataset(
        study_uids=train_df["StudyInstanceUID"].tolist(),
        study_dirs=train_dirs,
        labels=train_df[label_cols].values,
        is_gold=train_df["is_gold"].values,
        series_csv_path=series_csv or (PROJECT_ROOT / "data/raw/train_series.csv" if (PROJECT_ROOT / "data/raw/train_series.csv").is_file() else None),
        target_size=img_size,
        cached_tensor_dir=cached_tensor_dir,
        num_workers=num_workers,
    )
    val_ds = EfficiencyStudyDataset(
        study_uids=val_df["StudyInstanceUID"].tolist(),
        study_dirs=val_dirs,
        labels=val_df[label_cols].values,
        is_gold=val_df["is_gold"].values,
        series_csv_path=series_csv or (PROJECT_ROOT / "data/raw/train_series.csv" if (PROJECT_ROOT / "data/raw/train_series.csv").is_file() else None),
        target_size=img_size,
        cached_tensor_dir=cached_tensor_dir,
        num_workers=num_workers,
    )

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
        collate_fn=collate_efficiency,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=functools.partial(_seed_worker, base_seed=seed),
        pin_memory=(dev.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_efficiency,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=functools.partial(_seed_worker, base_seed=seed),
        pin_memory=(dev.type == "cuda"),
        drop_last=False,
    )

    # Compute label priors for prior-bias initialization
    pos_priors = np.clip(np.nan_to_num(np.nanmean(train_df[label_cols].values, axis=0), nan=0.15), 1e-4, 1.0 - 1e-4)

    if weights_path is not None:
        logger.info("Initializing DINOv2SlotHead with offline checkpoint: %s", weights_path)
    else:
        logger.warning("allow_random_init=True: DINOv2 backbone starts from RANDOM weights.")

    # Instantiate model
    model = DINOv2SlotHead(
        pretrained=False,
        weights_path=str(weights_path) if weights_path else None,
        img_size=img_size,
        init_bias_priors=pos_priors,
        slot_dropout_p=0.15,
    ).to(dev)

    # Loss function with continuous positive class weighting read from config
    raw_pos_weights = load_pos_weights(config_path)
    pos_weights_tensor = torch.as_tensor(raw_pos_weights, dtype=torch.float32, device=dev)
    criterion = MaskedBCEWithLogitsLoss(
        gold_weight=gold_weight,
        pos_weight=pos_weights_tensor,
        continuous_pos_weight=True,
    )

    # Optimizer and OneCycleLR scheduler
    optimizer = build_differential_optimizer(
        model=model,
        lr_backbone=lr_backbone,
        lr_head=lr_head,
        weight_decay=weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_loader) / max(1, grad_accum_steps))
    total_steps = max(1, steps_per_epoch * n_epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[lr_backbone, lr_backbone, lr_head, lr_head],
        total_steps=total_steps,
        pct_start=pct_start,
    )

    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=(use_amp and dev.type == "cuda"))

    top_checkpoints: list[tuple[float, int, Path]] = []
    history: list[dict[str, Any]] = []
    global_step = 0

    # Fixed gold study cohort for diagnostic attention heatmap logging (Spec §4.4)
    gold_indices = [i for i, g in enumerate(val_df["is_gold"].values) if g == 1][:8]
    if not gold_indices:
        gold_indices = list(range(min(4, len(val_ds))))
    sample_val_batch = collate_efficiency([val_ds[i] for i in gold_indices]) if len(val_ds) > 0 else None

    logger.info("Starting training fold %d on %s for %d epochs...", fold_id, dev, n_epochs)

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_samples = 0
        optimizer.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(train_loader):
            images = batch["image"].to(dev, non_blocking=True).float()
            masks = batch["presence_mask"].to(dev, non_blocking=True)
            targets = batch["labels"].to(dev, non_blocking=True)
            is_gold = batch["is_gold"].to(dev, non_blocking=True)

            with torch.amp.autocast(device_type=dev.type, dtype=torch.float16, enabled=(use_amp and dev.type == "cuda")):
                logits = model(images, masks)
                loss = criterion(logits.float(), targets, is_gold)
                if grad_accum_steps > 1:
                    loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            is_last_step = (step_idx + 1 == len(train_loader))
            if (step_idx + 1) % grad_accum_steps == 0 or is_last_step:
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            raw_loss_val = loss.item() * (grad_accum_steps if grad_accum_steps > 1 else 1.0)
            train_loss_total += raw_loss_val * len(targets)
            train_samples += len(targets)
            global_step += 1

            # Log step telemetry (Group 0 = backbone, Group 2 = head)
            current_lrs = {
                "backbone": optimizer.param_groups[0]["lr"],
                "slothead": optimizer.param_groups[2]["lr"],
            }
            logger_instance.log_step(
                step=global_step,
                loss=raw_loss_val,
                lrs=current_lrs,
            )

        epoch_train_loss = train_loss_total / max(1, train_samples)

        # Validation loop via reusable helper
        epoch_val_loss, concat_logits, concat_targets, concat_is_gold = run_validation(
            model=model,
            loader=val_loader,
            criterion=criterion,
            dev=dev,
            use_amp=use_amp,
        )

        # Calculate competition metrics
        metrics: MetricsResult
        if len(concat_logits) > 0:
            metrics = compute_competition_metrics(
                logits=concat_logits,
                targets=concat_targets,
                gold_mask=concat_is_gold.bool(),
                label_cols=label_cols,
            )
        else:
            metrics = MetricsResult(
                macro_auc_12=float("nan"),
                macro_auc_11=float("nan"),
                per_label_auc={},
                per_label_ap={},
                label_stats={},
                n_valid_labels=0,
                n_non_finite_logits=0,
            )

        # Log epoch telemetry
        logger_instance.log_epoch(
            epoch=epoch,
            train_loss=epoch_train_loss,
            val_loss=epoch_val_loss,
            metrics=metrics,
        )

        # Diagnostic visualizers: attention heatmap and PR curves
        if sample_val_batch is not None and (epoch == 1 or epoch == n_epochs or epoch % 3 == 0):
            with torch.no_grad():
                s_img = sample_val_batch["image"].to(dev).float()
                s_mask = sample_val_batch["presence_mask"].to(dev)
                _, s_attn = model.forward_with_attention(s_img, s_mask)
                logger_instance.log_attention_heatmap(
                    attn_weights=s_attn.float().cpu().numpy(),
                    step=epoch,
                    presence_mask=s_mask.cpu().numpy(),
                    slot_names=SLOT_NAMES,
                    label_cols=label_cols,
                )
                if len(concat_logits) > 0:
                    logger_instance.log_pr_curves(
                        targets=concat_targets,
                        logits=concat_logits,
                        step=epoch,
                        gold_mask=concat_is_gold.bool(),
                        label_cols=label_cols,
                    )

        # Rank checkpoints by macro_auc_11 (large n, stable across epochs)
        score = float(metrics.macro_auc_11)
        if np.isnan(score):
            score = -float(epoch_val_loss)

        ckpt_file = ckpt_dir / f"checkpoint_epoch_{epoch:02d}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "macro_auc_12": float(metrics.macro_auc_12),
                "macro_auc_11": float(metrics.macro_auc_11),
                "val_loss": float(epoch_val_loss),
            },
            ckpt_file,
        )

        _register_checkpoint(top_checkpoints, score, epoch, ckpt_file, k=3)

        history.append({
            "epoch": epoch,
            "train_loss": float(epoch_train_loss),
            "val_loss": float(epoch_val_loss),
            "macro_auc_11": float(metrics.macro_auc_11),
            "macro_auc_12": float(metrics.macro_auc_12),
            "score": float(score),
        })

        logger.info(
            "Epoch %02d/%02d | Train Loss: %.4f | Val Loss: %.4f | Macro-12: %.4f | Macro-11: %.4f | Score: %.4f",
            epoch,
            n_epochs,
            epoch_train_loss,
            epoch_val_loss,
            metrics.macro_auc_12,
            metrics.macro_auc_11,
            score,
        )

    # Post-training: average top-3 checkpoint weights into final ensemble file
    top3_paths = [item[2] for item in top_checkpoints[:3]]
    ensemble_path = ckpt_dir / "dinov2_slothead_top3.pth"
    average_top_checkpoints(top3_paths, ensemble_path)

    # Evaluate the averaged ensemble model
    averaged_metrics: MetricsResult | None = None
    if len(val_loader) > 0 and ensemble_path.is_file():
        model.load_state_dict(torch.load(ensemble_path, map_location=dev, weights_only=True))
        _, avg_logits, avg_targets, avg_gold = run_validation(
            model=model, loader=val_loader, criterion=criterion, dev=dev, use_amp=use_amp
        )
        averaged_metrics = compute_competition_metrics(
            logits=avg_logits,
            targets=avg_targets,
            gold_mask=avg_gold.bool(),
            label_cols=label_cols,
        )
        logger.info(
            "Averaged Top-3 Model Validation | Macro-12: %.4f | Macro-11: %.4f",
            averaged_metrics.macro_auc_12,
            averaged_metrics.macro_auc_11,
        )

    best_single = max(history, key=lambda h: h["score"])
    delta = None
    recommended = ensemble_path
    if (
        averaged_metrics is not None
        and not np.isnan(averaged_metrics.macro_auc_11)
        and not np.isnan(best_single["macro_auc_11"])
    ):
        delta = float(averaged_metrics.macro_auc_11) - best_single["macro_auc_11"]
        if delta < -0.005:
            logger.warning(
                "Averaged top-3 is %.4f worse than best single epoch; recommending best single.",
                -delta,
            )
            best_single_path = ckpt_dir / "dinov2_slothead_best_single.pth"
            best_ckpt_data = torch.load(top3_paths[0], map_location="cpu", weights_only=True)
            torch.save(best_ckpt_data["model_state_dict"], best_single_path)
            recommended = best_single_path

    logger_instance.flush()
    logger_instance.close()

    return {
        "best_score": top_checkpoints[0][0] if top_checkpoints else 0.0,
        "best_epoch": top_checkpoints[0][1] if top_checkpoints else 0,
        "top3_checkpoints": [str(p) for p in top3_paths],
        "averaged_weights_path": str(ensemble_path),
        "recommended_weights_path": str(recommended),
        "averaged_vs_best_single_delta": delta,
        "averaged_macro_auc_11": float(averaged_metrics.macro_auc_11) if averaged_metrics else None,
        "averaged_macro_auc_12": float(averaged_metrics.macro_auc_12) if averaged_metrics else None,
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI training arguments."""
    parser = argparse.ArgumentParser(description="RSNA Knee MRI Single-GPU Training Pipeline")
    parser.add_argument("--fold", type=int, default=0, help="Validation fold ID (0, 1, 2)")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Studies per batch")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr-backbone", type=float, default=3e-5, help="Backbone peak learning rate")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="SlotHead peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.02, help="AdamW weight decay")
    parser.add_argument("--img-size", type=int, default=336, help="Image slice resolution (default: 336)")
    parser.add_argument("--weights-path", type=str, default=None, help="Offline DINOv2 backbone checkpoint path")
    parser.add_argument("--allow-random-init", action="store_true", help="Permit training without backbone weights (tests only)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader background worker count")
    parser.add_argument("--limit-train", type=int, default=None, help="Subset size for training dry runs")
    parser.add_argument("--limit-val", type=int, default=None, help="Subset size for validation dry runs")
    parser.add_argument("--device", type=str, default="cuda:0", help="Active device (cuda:0 or cpu)")
    parser.add_argument("--config-path", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--labels-path", type=str, default=None, help="Path to soft labels")
    parser.add_argument("--splits-path", type=str, default=None, help="Path to CV splits")
    parser.add_argument("--dicom-root", type=str, default=None, help="DICOM directory root")
    parser.add_argument("--cache-dir", type=str, default=None, help="Pre-extracted tensor cache directory")
    parser.add_argument("--weights-dir", type=str, default=None, help="Checkpoint output directory")
    parser.add_argument("--tb-dir", type=str, default=None, help="TensorBoard output directory")
    parser.add_argument("--no-amp", action="store_true", help="Disable Automatic Mixed Precision")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    train_efficiency_fold(
        fold_id=args.fold,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        img_size=args.img_size,
        weights_path=args.weights_path,
        allow_random_init=args.allow_random_init,
        num_workers=args.num_workers,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        config_path=args.config_path,
        device=args.device,
        labels_path=args.labels_path,
        splits_path=args.splits_path,
        dicom_root=args.dicom_root,
        cached_tensor_dir=args.cache_dir,
        weights_dir=args.weights_dir,
        tb_log_dir=args.tb_dir,
        use_amp=not args.no_amp,
    )
