"""
trainer.py -- Training Loop for RSNA Knee MRI Phase 3 Baseline

Features:
    - 5-fold StratifiedGroupKFold cross-validation
    - KneeMRIDatasetCached (fast .npz loading) with KneeMRIDataset fallback
    - plane_present mask propagated to KneeMILModel.forward() every step
    - Gradient accumulation (batch x accum_steps -> effective batch 8)
    - Per-label + macro AUC (NaN-masked) logged every epoch
    - Per-epoch wall time and ETA printed
    - Best checkpoint by val macro AUC per fold (saved to outputs/weights/)
    - Backbone warm-up: frozen for freeze_backbone_epochs, then unfrozen
    - OOF gold logits collected from best-epoch validation set
      -> returned by train_fold() for post-training Platt calibration
    - AMP (Automatic Mixed Precision) with GradScaler
    - num_workers=2: safe with local cache (not network FUSE mount)
    - Phase D budget gate: pre-flight ETA check before committing to full run
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.datasets.mri_dataset import KneeMRIDataset, KneeMRIDatasetCached
from src.models.mil_model import KneeMILModel
from src.training.losses import MaskedBCEWithLogitsLoss


def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility across CPU, GPU, numpy, Python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_per_label_auc(
    all_logits: np.ndarray,
    all_targets: np.ndarray,
    label_cols: list[str],
) -> dict[str, float]:
    """Compute per-label and macro AUC, excluding NaN-masked samples per label.

    Applies sigmoid to convert raw logits to probabilities before AUC computation.
    NaN targets are excluded per-label (different labels may have different
    subsets of valid studies).

    Args:
        all_logits: (N, 12) raw logits -- NOT sigmoid-activated.
        all_targets: (N, 12) float targets -- NaN for unaddressed silence.
        label_cols: 12 label names from config.yaml labels.columns.

    Returns:
        Dict with per-label AUC floats and 'macro_auc' (mean of valid-label AUCs).
    """
    all_sigmoid = 1.0 / (1.0 + np.exp(-all_logits))
    scores: dict[str, float] = {}

    for i, col in enumerate(label_cols):
        valid = ~np.isnan(all_targets[:, i])
        if valid.sum() < 2:
            scores[col] = float("nan")
            continue
        try:
            scores[col] = roc_auc_score(
                all_targets[valid, i], all_sigmoid[valid, i]
            )
        except ValueError:
            scores[col] = float("nan")

    valid_aucs = [v for v in scores.values() if not np.isnan(v)]
    scores["macro_auc"] = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    return scores


def collate_fn(batch: list[dict]) -> dict[str, Any]:
    """Custom collate function that preserves NaN in label tensors."""
    return {
        "sagittal":      torch.stack([b["sagittal"]      for b in batch]),
        "coronal":       torch.stack([b["coronal"]       for b in batch]),
        "axial":         torch.stack([b["axial"]         for b in batch]),
        "plane_present": torch.stack([b["plane_present"] for b in batch]),
        "labels":        torch.stack([b["labels"]        for b in batch]),
        "is_gold":       torch.tensor(
            [b["is_gold"] for b in batch], dtype=torch.long
        ),
        "study_uid":     [b["study_uid"] for b in batch],
    }


def _make_augment_fn():
    """Build albumentations augmentation pipeline for training."""
    try:
        import albumentations as A
        fn = A.Compose([
            A.RandomRotate90(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1,
                rotate_limit=15, p=0.5
            ),
            A.GaussNoise(var_limit=(0.001, 0.005), p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=0.3
            ),
        ])
        print("   Augmentation: albumentations pipeline active")
        return fn
    except ImportError:
        print("   albumentations not installed -- training without augmentation")
        return None


def run_budget_gate(
    train_loader: "DataLoader",
    n_epochs: int,
    n_folds: int,
    budget_hours: float = 8.5,
    n_probe_batches: int = 3,
) -> None:
    """Phase D budget gate: time real DataLoader batches and project total runtime.

    Times n_probe_batches actual batches from train_loader (not a mock), projects
    the total runtime for all folds and epochs, and prints a warning if the
    projection exceeds the session budget.

    Does NOT raise or abort -- prints a warning so the user can decide.
    Call this at the start of fold 0 before the training loop begins.

    Args:
        train_loader:    The real DataLoader (cache-backed or DICOM-backed).
        n_epochs:        Epochs per fold.
        n_folds:         Total folds (for full run projection).
        budget_hours:    Budget limit in hours. Default 8.5 for Kaggle's 9h cap.
        n_probe_batches: Number of batches to time for projection. Default 3.
    """
    print("\n   [Budget Gate] Timing real DataLoader batches...")
    sys.stdout.flush()

    times = []
    for i, _ in enumerate(train_loader):
        if i == 0:
            t_start = time.time()  # Start after first batch (warms up workers)
        elif i <= n_probe_batches:
            times.append(time.time() - t_start)
            t_start = time.time()
        if i >= n_probe_batches:
            break

    if not times:
        print("   [Budget Gate] Could not time batches (DataLoader empty?). Skipping.")
        return

    secs_per_batch  = sum(times) / len(times)
    n_batches_epoch = len(train_loader)
    secs_per_epoch  = secs_per_batch * n_batches_epoch
    projected_total = secs_per_epoch * n_epochs * n_folds
    projected_hours = projected_total / 3600.0

    print(
        f"   [Budget Gate] ~{secs_per_batch:.2f}s/batch x "
        f"{n_batches_epoch} batches x "
        f"{n_epochs} epochs x "
        f"{n_folds} folds = "
        f"~{projected_hours:.2f}h projected"
    )

    if projected_hours > budget_hours:
        print(
            f"   [Budget Gate] WARNING: Projected {projected_hours:.2f}h "
            f"exceeds budget of {budget_hours:.1f}h. "
            "Consider reducing n_epochs, n_slices, or batch_size before launching."
        )
    else:
        print(
            f"   [Budget Gate] OK: {projected_hours:.2f}h < {budget_hours:.1f}h budget."
        )
    sys.stdout.flush()


def train_fold(
    fold_id: int,
    folds_df: pd.DataFrame,
    pseudo_labels_df: pd.DataFrame,
    label_cols: list[str],
    weights_dir: str,
    logs_dir: str,
    cfg: dict,
    cache_dir: Optional[str] = None,
    dicom_root: Optional[str] = None,
) -> dict:
    """Train one fold of the 5-fold StratifiedGroupKFold cross-validation.

    Uses KneeMRIDatasetCached (fast) if cache_dir is provided,
    falls back to KneeMRIDataset (slow live DICOM) otherwise.

    Args:
        fold_id:          Validation fold index (0-4).
        folds_df:         cv_folds_5fold.csv DataFrame.
        pseudo_labels_df: pseudo_labels.csv DataFrame (4,407 rows).
        label_cols:       12 label column names from config.yaml.
        weights_dir:      Directory for per-fold model checkpoints.
        logs_dir:         Directory for training logs.
        cfg:              Full config.yaml dict.
        cache_dir:        Path to .npz cache dir (fast loader). None = use DICOM.
        dicom_root:       DICOM data root path (used as fallback if cache_dir set,
                          or primary if cache_dir is None).

    Returns:
        dict with:
            'auc_scores':       dict of best val AUCs per label + macro_auc
            'oof_gold_logits':  np.ndarray (n_gold_in_fold, 12) OOF logits
            'oof_gold_targets': np.ndarray (n_gold_in_fold, 12) OOF targets
    """
    set_seed(cfg["project"]["seed"])
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accum_steps = cfg["training"].get("grad_accum_steps", 1)
    eff_batch   = cfg["training"]["batch_size"] * accum_steps

    print(f"\n{'='*60}")
    print(f"  FOLD {fold_id} | Device: {device} | Effective batch: {eff_batch}")
    print(f"  Cache mode: {'FAST (.npz)' if cache_dir else 'SLOW (live DICOM)'}")
    print(f"{'='*60}")
    sys.stdout.flush()

    # Build train/val DataFrames
    fold_meta  = folds_df[["StudyInstanceUID", "is_gold"]].drop_duplicates()
    val_uids   = set(folds_df[folds_df["fold_id"] == fold_id]["StudyInstanceUID"])
    train_uids = set(folds_df[folds_df["fold_id"] != fold_id]["StudyInstanceUID"])

    train_df = (
        pseudo_labels_df[pseudo_labels_df["StudyInstanceUID"].isin(train_uids)]
        .copy()
        .merge(fold_meta, on="StudyInstanceUID", how="left")
    )
    val_df = (
        pseudo_labels_df[pseudo_labels_df["StudyInstanceUID"].isin(val_uids)]
        .copy()
        .merge(fold_meta, on="StudyInstanceUID", how="left")
    )

    print(f"   Train: {len(train_df)} studies | Val: {len(val_df)} studies")
    print(f"   Gold in train: {int(train_df['is_gold'].sum())} | "
          f"Gold in val: {int(val_df['is_gold'].sum())}")
    sys.stdout.flush()

    augment_fn = _make_augment_fn()

    # Select dataset class based on cache availability
    n_slices    = cfg["model"]["n_slices"]
    target_size = tuple(cfg["model"]["target_size"])
    stack_size  = cfg["model"]["stack_size"]
    n_workers   = 2 if cache_dir else 0  # Local cache is safe for multi-worker

    if cache_dir:
        def _make_dataset(df: pd.DataFrame, is_train: bool) -> KneeMRIDatasetCached:
            return KneeMRIDatasetCached(
                study_df=df,
                cache_dir=cache_dir,
                label_cols=label_cols,
                n_slices=n_slices,
                target_size=target_size,
                stack_size=stack_size,
                is_train=is_train,
                augment_fn=augment_fn if is_train else None,
                dicom_root=dicom_root,
            )
    else:
        def _make_dataset(df: pd.DataFrame, is_train: bool) -> KneeMRIDataset:
            return KneeMRIDataset(
                study_df=df,
                dicom_root=dicom_root,
                label_cols=label_cols,
                n_slices=n_slices,
                target_size=target_size,
                stack_size=stack_size,
                is_train=is_train,
                augment_fn=augment_fn if is_train else None,
            )

    train_loader = DataLoader(
        _make_dataset(train_df, is_train=True),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=n_workers,
        pin_memory=(n_workers > 0),  # pin_memory only useful with workers
        persistent_workers=(n_workers > 0),
        collate_fn=collate_fn,
        drop_last=False,  # Keep all studies per epoch; BatchNorm safe with batch>=2
    )
    val_loader = DataLoader(
        _make_dataset(val_df, is_train=False),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=n_workers,
        pin_memory=(n_workers > 0),
        persistent_workers=(n_workers > 0),
        collate_fn=collate_fn,
    )

    # Phase D budget gate -- run once at fold 0 to confirm we fit within session
    if fold_id == 0:
        run_budget_gate(
            train_loader=train_loader,
            n_epochs=cfg["training"]["n_epochs"],
            n_folds=cfg["cv"]["n_folds"],
            budget_hours=8.5,
        )


    model = KneeMILModel(
        backbone_name=cfg["model"]["backbone"],
        n_classes=len(label_cols),
        stack_size=stack_size,
        pretrained=True,
        local_weights_path=cfg["model"].get("local_weights_path"),
        dropout=cfg["training"]["dropout"],
        freeze_backbone_epochs=cfg["training"]["freeze_backbone_epochs"],
        use_grad_checkpointing=cfg["training"].get("use_grad_checkpointing", True),
    ).to(device)

    criterion = MaskedBCEWithLogitsLoss(
        gold_weight=cfg["training"]["gold_weight"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["n_epochs"],
        eta_min=cfg["training"]["lr"] * 0.01,
    )
    scaler = GradScaler("cuda")

    best_macro_auc = 0.0
    best_auc_scores: dict[str, float] = {}
    oof_gold_logits_best:  list[np.ndarray] = []
    oof_gold_targets_best: list[np.ndarray] = []

    Path(weights_dir).mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(weights_dir) / f"fold{fold_id}_best.pth"

    fold_t_start = time.time()

    for epoch in range(cfg["training"]["n_epochs"]):
        epoch_t_start = time.time()
        model.on_epoch_start(epoch)

        # ---- Train ----
        model.train()
        train_loss, n_steps = 0.0, 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            planes = {
                k: batch[k].to(device)
                for k in ["sagittal", "coronal", "axial"]
            }
            plane_mask = batch["plane_present"].to(device)
            labels     = batch["labels"].to(device)
            is_gold    = batch["is_gold"].to(device)

            with autocast("cuda"):
                logits, _ = model(planes, plane_mask=plane_mask)
                loss = criterion(logits, labels, is_gold) / accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg["training"]["grad_clip_norm"]
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accum_steps
            n_steps    += 1

        scheduler.step()
        avg_train_loss = train_loss / max(n_steps, 1)

        # ---- Validate ----
        model.eval()
        all_logits_list:  list[np.ndarray] = []
        all_targets_list: list[np.ndarray] = []
        gold_logits_ep:   list[np.ndarray] = []
        gold_targets_ep:  list[np.ndarray] = []

        with torch.no_grad():
            for batch in val_loader:
                planes = {
                    k: batch[k].to(device)
                    for k in ["sagittal", "coronal", "axial"]
                }
                plane_mask   = batch["plane_present"].to(device)
                labels       = batch["labels"]
                is_gold_mask = batch["is_gold"].numpy().astype(bool)

                with autocast("cuda"):
                    logits, _ = model(planes, plane_mask=plane_mask)

                lnp = logits.cpu().numpy()
                tnp = labels.numpy()

                all_logits_list.append(lnp)
                all_targets_list.append(tnp)

                if is_gold_mask.any():
                    gold_logits_ep.append(lnp[is_gold_mask])
                    gold_targets_ep.append(tnp[is_gold_mask])

        all_logits  = np.concatenate(all_logits_list,  axis=0)
        all_targets = np.concatenate(all_targets_list, axis=0)
        auc_scores  = compute_per_label_auc(all_logits, all_targets, label_cols)
        macro_auc   = auc_scores["macro_auc"]

        # Epoch timing and ETA
        epoch_elapsed = time.time() - epoch_t_start
        fold_elapsed  = time.time() - fold_t_start
        epochs_done   = epoch + 1
        epochs_left   = cfg["training"]["n_epochs"] - epochs_done
        eta_fold      = fold_elapsed / epochs_done * epochs_left
        n_folds_total = cfg["cv"]["n_folds"]
        folds_left    = n_folds_total - fold_id - 1
        eta_total     = (fold_elapsed / epochs_done) * (
            epochs_left + cfg["training"]["n_epochs"] * folds_left
        )

        print(
            f"   Ep {epoch+1:3d}/{cfg['training']['n_epochs']} | "
            f"Loss: {avg_train_loss:.4f} | MacroAUC: {macro_auc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"EpTime: {epoch_elapsed/60:.1f}m | "
            f"ETA fold: {eta_fold/60:.1f}m | ETA total: {eta_total/3600:.2f}h"
        )
        for col in label_cols:
            auc = auc_scores.get(col, float("nan"))
            auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
            print(f"      {col:25s}: {auc_str}")
        sys.stdout.flush()

        # Save best checkpoint
        if macro_auc > best_macro_auc:
            best_macro_auc        = macro_auc
            best_auc_scores       = auc_scores
            oof_gold_logits_best  = gold_logits_ep
            oof_gold_targets_best = gold_targets_ep

            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "macro_auc":        macro_auc,
                "auc_scores":       auc_scores,
                "fold_id":          fold_id,
                "cfg":              cfg,
            }, checkpoint_path)
            print(f"    Checkpoint saved: {checkpoint_path} (AUC={macro_auc:.4f})")
            sys.stdout.flush()

    oof_logits = (
        np.concatenate(oof_gold_logits_best, axis=0)
        if oof_gold_logits_best else np.empty((0, len(label_cols)))
    )
    oof_targets = (
        np.concatenate(oof_gold_targets_best, axis=0)
        if oof_gold_targets_best else np.empty((0, len(label_cols)))
    )

    total_fold_time = time.time() - fold_t_start
    print(f"\n   Fold {fold_id} complete in {total_fold_time/60:.1f} min. "
          f"Best Macro AUC: {best_macro_auc:.4f}")
    print(f"   OOF gold studies collected: {len(oof_logits)}")
    sys.stdout.flush()

    return {
        "auc_scores":       best_auc_scores,
        "oof_gold_logits":  oof_logits,
        "oof_gold_targets": oof_targets,
    }
