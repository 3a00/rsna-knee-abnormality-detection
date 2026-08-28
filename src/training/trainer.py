"""
trainer.py -- Training Loop for RSNA Knee MRI Phase 3 Baseline

Features:
    - 5-fold StratifiedGroupKFold cross-validation
    - plane_present mask propagated to KneeMILModel.forward() every step
    - Gradient accumulation (batch=4, accum=2 -> effective batch=8 for VRAM)
    - Per-label + macro AUC (NaN-masked per label) logged every epoch
    - Best checkpoint by val macro AUC per fold
    - Backbone warm-up: frozen for freeze_backbone_epochs, then unfrozen
    - OOF gold logits collected from best-epoch validation set
      -> returned by train_fold() for post-training Platt calibration
    - AMP (Automatic Mixed Precision) with GradScaler
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.datasets.mri_dataset import KneeMRIDataset
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
    # Sigmoid to probabilities (only for AUC -- never for loss)
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
            # Only one class in val -- skip label
            scores[col] = float("nan")

    valid_aucs = [v for v in scores.values() if not np.isnan(v)]
    scores["macro_auc"] = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    return scores


def collate_fn(batch: list[dict]) -> dict[str, Any]:
    """Custom collate function that preserves NaN in label tensors.

    Default torch.utils.data.default_collate converts NaN to 0 in some cases.
    This implementation explicitly stacks all keys and preserves NaN values.
    """
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


def train_fold(
    fold_id: int,
    folds_df: pd.DataFrame,
    pseudo_labels_df: pd.DataFrame,
    dicom_root: str,
    label_cols: list[str],
    weights_dir: str,
    logs_dir: str,
    cfg: dict,
) -> dict:
    """Train one fold of the 5-fold StratifiedGroupKFold cross-validation.

    Args:
        fold_id: Validation fold index (0-4).
        folds_df: cv_folds_5fold.csv DataFrame with fold_id, is_gold columns.
        pseudo_labels_df: pseudo_labels.csv DataFrame (4,407 rows).
        dicom_root: DICOM data root path.
        label_cols: 12 label column names from config.yaml.
        weights_dir: Directory for per-fold model checkpoints.
        logs_dir: Directory for training logs.
        cfg: Full config.yaml dict.

    Returns:
        dict with:
            'auc_scores':        dict of best val AUCs per label + macro_auc
            'oof_gold_logits':   np.ndarray (n_gold_in_fold, 12) OOF logits
            'oof_gold_targets':  np.ndarray (n_gold_in_fold, 12) OOF targets
    """
    set_seed(cfg["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accum_steps = cfg["training"].get("grad_accum_steps", 1)
    eff_batch   = cfg["training"]["batch_size"] * accum_steps

    print(f"\n{'='*60}")
    print(f"  FOLD {fold_id} | Device: {device} | Effective batch: {eff_batch}")
    print(f"{'='*60}")

    # Build train/val DataFrames from fold assignment
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
    print(f"   Gold in train: {train_df['is_gold'].sum()} | "
          f"Gold in val: {val_df['is_gold'].sum()}")

    # Build augmentation pipeline
    try:
        import albumentations as A
        augment_fn = A.Compose([
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
    except ImportError:
        augment_fn = None
        print("     albumentations not installed -- training without augmentation")

    def _make_dataset(df: pd.DataFrame, is_train: bool) -> KneeMRIDataset:
        return KneeMRIDataset(
            study_df=df,
            dicom_root=dicom_root,
            label_cols=label_cols,
            n_slices=cfg["model"]["n_slices"],
            target_size=tuple(cfg["model"]["target_size"]),
            stack_size=cfg["model"]["stack_size"],
            is_train=is_train,
            augment_fn=augment_fn if is_train else None,
        )

    # num_workers=0: synchronous loading — prevents multiprocessing deadlock
    # on Kaggle's network-mounted DICOM storage. pin_memory=False pairs with this.
    train_loader = DataLoader(
        _make_dataset(train_df, is_train=True),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
        drop_last=True,  # Avoid single-sample batches with BatchNorm
    )
    val_loader = DataLoader(
        _make_dataset(val_df, is_train=False),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    # Initialize model
    model = KneeMILModel(
        backbone_name=cfg["model"]["backbone"],
        n_classes=len(label_cols),
        stack_size=cfg["model"]["stack_size"],
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

    for epoch in range(cfg["training"]["n_epochs"]):
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
                # Divide by accum_steps to simulate larger batch
                loss = criterion(logits, labels, is_gold) / accum_steps

            scaler.scale(loss).backward()

            # Optimizer step every accum_steps batches
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

                # Collect OOF logits for gold studies in this fold's val set
                if is_gold_mask.any():
                    gold_logits_ep.append(lnp[is_gold_mask])
                    gold_targets_ep.append(tnp[is_gold_mask])

        all_logits  = np.concatenate(all_logits_list,  axis=0)
        all_targets = np.concatenate(all_targets_list, axis=0)
        auc_scores  = compute_per_label_auc(all_logits, all_targets, label_cols)
        macro_auc   = auc_scores["macro_auc"]

        # Log epoch results
        import sys
        print(
            f"   Ep {epoch+1:3d}/{cfg['training']['n_epochs']} | "
            f"Loss: {avg_train_loss:.4f} | MacroAUC: {macro_auc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )
        for col in label_cols:
            auc = auc_scores.get(col, float("nan"))
            auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
            print(f"      {col:25s}: {auc_str}")
        sys.stdout.flush()

        # Save best checkpoint and OOF gold logits
        if macro_auc > best_macro_auc:
            best_macro_auc   = macro_auc
            best_auc_scores  = auc_scores
            # Store OOF gold logits from this best-epoch validation pass
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

    # Aggregate OOF gold logits for Platt calibration
    oof_logits = (
        np.concatenate(oof_gold_logits_best, axis=0)
        if oof_gold_logits_best else np.empty((0, len(label_cols)))
    )
    oof_targets = (
        np.concatenate(oof_gold_targets_best, axis=0)
        if oof_gold_targets_best else np.empty((0, len(label_cols)))
    )

    print(f"\n   Fold {fold_id} complete. Best Macro AUC: {best_macro_auc:.4f}")
    print(f"   OOF gold studies collected: {len(oof_logits)}")

    return {
        "auc_scores":        best_auc_scores,
        "oof_gold_logits":   oof_logits,
        "oof_gold_targets":  oof_targets,
    }
