"""
mri_dataset.py -- PyTorch Dataset for RSNA Knee MRI Studies (v3 Final)

Key design decisions:
    - Augmentation applied AFTER stack_25d on (H,W,3) stacked tensor so all
      3 adjacent channels receive identical spatial transforms (2.5D alignment).
    - synovitis_soft used as float BCE target for non-gold rows.
    - Bilateral studies: keep-all strategy (no laterality filtering) until
      a confirmed per-study laterality signal is available. Warning logged.
    - plane_present bool tensor returned per study for model plane masking.
    - None-safe series_description: (series.series_description or "").upper()

Output convention:
    labels tensor preserves NaN for unaddressed silence -- never replace with 0.
    MaskedBCEWithLogitsLoss handles NaN masking at training time.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.datasets.dicom_loader import (
    AnatomicalPlane,
    DICOMSeries,
    SeriesType,
    load_study_series,
)


PLANE_ORDER     = [AnatomicalPlane.SAGITTAL, AnatomicalPlane.CORONAL, AnatomicalPlane.AXIAL]
SERIES_PRIORITY = [SeriesType.FLUID_SENSITIVE, SeriesType.ANATOMICAL]

# Label column names for Synovitis routing
SYNOVITIS_COL      = "Synovitis"
SYNOVITIS_SOFT_COL = "synovitis_soft"


# ============================================================================
# Slice Selection
# ============================================================================

def select_slices(
    volume: np.ndarray,
    n_slices: int,
    center_weight: float = 2.0,
) -> np.ndarray:
    """Select N representative slices with center-weighted uniform sampling.

    The central 50% of the volume (joint articular space) is sampled at
    2x density relative to the periphery. Ensures the joint line -- where
    meniscal tears, cartilage loss, and effusion are most visible -- is
    proportionally more represented than the peri-articular bone.

    Volumes smaller than n_slices are zero-padded at the end.

    Args:
        volume: (n_slices, H, W) float32 MRI volume, values in [0, 1].
        n_slices: Target number of slices to select.
        center_weight: Density multiplier for central 50% of volume.

    Returns:
        (n_slices, H, W) float32 array.
    """
    total = volume.shape[0]

    if total <= n_slices:
        pad = np.zeros((n_slices - total, *volume.shape[1:]), dtype=volume.dtype)
        return np.concatenate([volume, pad], axis=0)

    weights = np.ones(total, dtype=np.float32)
    weights[int(total * 0.25):int(total * 0.75)] = center_weight
    weights /= weights.sum()

    rng = np.random.default_rng(seed=42)
    indices = np.sort(rng.choice(total, size=n_slices, replace=False, p=weights))
    return volume[indices]


# ============================================================================
# 2.5D Stacking
# ============================================================================

def stack_25d(volume: np.ndarray, stack_size: int = 3) -> np.ndarray:
    """Stack consecutive slices as 2.5D channels: (N, H, W) -> (N, stack_size, H, W).

    Each slice i gets channels: [slice i-half, ..., slice i, ..., slice i+half].
    Boundaries clamp to edge (no wrap-around). stack_size must be odd.

    Satisfies the clinical 2-slice meniscal tear rule: the model sees adjacent
    slices simultaneously in a single backbone forward pass, enabling detection
    of abnormalities that span multiple consecutive MRI slices.

    Args:
        volume: (N, H, W) float32 slice volume.
        stack_size: Must be odd (e.g., 3 -> prev, current, next).

    Returns:
        (N, stack_size, H, W) float32 array.
    """
    assert stack_size % 2 == 1, f"stack_size must be odd, got {stack_size}"
    half = stack_size // 2
    n = volume.shape[0]
    stacked = np.zeros((n, stack_size, *volume.shape[1:]), dtype=volume.dtype)

    for i in range(n):
        for j, offset in enumerate(range(-half, half + 1)):
            stacked[i, j] = volume[np.clip(i + offset, 0, n - 1)]

    return stacked


# ============================================================================
# Main Dataset Class
# ============================================================================

class KneeMRIDataset(Dataset):
    """PyTorch Dataset for RSNA Knee Abnormality Detection (v3 Final).

    Loads and preprocesses multi-series, multi-plane knee MRI studies for
    the 2.5D MIL model. Handles the three-state label scheme:
        1   -> included, target=1.0
        0   -> included, target=0.0
        NaN -> excluded (silence / unaddressed) -- preserved for BCE masking

    Args:
        study_df: DataFrame with StudyInstanceUID, label columns, is_gold.
        dicom_root: DICOM root path (Kaggle: /kaggle/input/...).
        label_cols: 12 label column names from config.yaml labels.columns.
        n_slices: Slices to select per series (default: 24).
        target_size: Resize each slice to (H, W) (default: (224, 224)).
        stack_size: 2.5D stacking depth -- must be odd (default: 3).
        is_train: Apply data augmentation if True.
        augment_fn: Optional albumentations Compose transform.
                    Applied AFTER stack_25d on (H, W, stack_size) so all
                    channels share identical spatial transforms.

    Returns per __getitem__:
        {
          "sagittal":      Tensor (N, stack_size, H, W) float32,
          "coronal":       Tensor (N, stack_size, H, W) float32,
          "axial":         Tensor (N, stack_size, H, W) float32,
          "plane_present": Tensor (3,) bool  -- [sag_ok, cor_ok, axial_ok],
          "labels":        Tensor (12,) float32 -- NaN for masked labels,
          "is_gold":       int (0 or 1),
          "study_uid":     str,
        }
    """

    def __init__(
        self,
        study_df: pd.DataFrame,
        dicom_root: str,
        label_cols: list[str],
        n_slices: int = 24,
        target_size: tuple[int, int] = (224, 224),
        stack_size: int = 3,
        is_train: bool = True,
        augment_fn=None,
    ) -> None:
        self.study_df   = study_df.reset_index(drop=True)
        self.dicom_root = Path(dicom_root)
        self.label_cols = label_cols
        self.n_slices   = n_slices
        self.target_size = target_size
        self.stack_size  = stack_size
        self.is_train    = is_train
        self.augment_fn  = augment_fn

    def __len__(self) -> int:
        return len(self.study_df)

    def _resize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Resize each slice to target_size using bilinear interpolation."""
        import cv2
        H, W = self.target_size
        return np.stack([
            cv2.resize(sl, (W, H), interpolation=cv2.INTER_LINEAR)
            for sl in volume
        ]).astype(np.float32)

    def _check_bilateral(self, series_list: list[DICOMSeries], study_uid: str) -> None:
        """Log a warning for likely bilateral studies; do NOT filter series.

        Safe keep-all strategy: bilateral studies keep all their series.
        The model processes both knees together via attention pooling --
        suboptimal but not harmful compared to guessing laterality wrongly.

        Long-term path (Phase 4+): extract ImageLaterality from DICOM headers
        during scan_dicom_fingerprints.py, store per-study laterality in
        dicom_fingerprints.csv, pass target_laterality to the dataset.
        DO NOT hardcode 'L' or 'R' -- always derive from data.
        """
        if len(series_list) > 12:
            warnings.warn(
                f"Study {study_uid} may be bilateral ({len(series_list)} series). "
                "Keeping all series (safe default -- no laterality filtering). "
                "To enable routing, add ImageLaterality to scan_dicom_fingerprints.py.",
                UserWarning,
                stacklevel=2,
            )

    def _select_best_series(
        self,
        series_list: list[DICOMSeries],
        plane: AnatomicalPlane,
    ) -> Optional[DICOMSeries]:
        """Select fluid-sensitive series for given plane; fallback to anatomical.

        Uses (series.series_description or "").upper() defensively for None descriptions.

        Priority:
            1. Fluid-sensitive (T2/PD-FS/STIR) -- best for pathology detection
            2. Anatomical (T1) -- fallback for plane coverage
            3. Most slices if multiple candidates in same type
        """
        candidates = [s for s in series_list if s.plane == plane]
        if not candidates:
            return None

        for stype in SERIES_PRIORITY:
            typed = [s for s in candidates if s.series_type == stype]
            if typed:
                return max(typed, key=lambda s: s.n_slices)

        # Fallback: most slices regardless of type
        return max(candidates, key=lambda s: s.n_slices)

    def _load_plane_tensor(
        self,
        series_list: list[DICOMSeries],
        plane: AnatomicalPlane,
    ) -> tuple[torch.Tensor, bool]:
        """Load, select, stack, augment, and tensorize slices for one plane.

        Augmentation order (critical for 2.5D alignment):
            1. resize_volume  -> (n_slices, H, W)      per-series resize
            2. select_slices  -> (N, H, W)             center-weighted selection
            3. stack_25d      -> (N, stack_size, H, W) 2.5D channel stacking
            4. augment_fn     -> applied to (H, W, stack_size) per stack-slice
                                 so ALL 3 channels get identical spatial transform

        Returns:
            (tensor (N, stack_size, H, W), plane_present: bool)
            plane_present=False if no series found for this plane.
        """
        series = self._select_best_series(series_list, plane)

        if series is None:
            return (
                torch.zeros(
                    self.n_slices, self.stack_size, *self.target_size,
                    dtype=torch.float32
                ),
                False,
            )

        volume  = self._resize_volume(series.pixel_array)  # (n_slices, H, W)
        volume  = select_slices(volume, self.n_slices)     # (N, H, W)
        stacked = stack_25d(volume, self.stack_size)        # (N, stack_size, H, W)

        # Augment AFTER stacking -- all channels share the same spatial transform
        if self.is_train and self.augment_fn is not None:
            augmented = []
            for i in range(self.n_slices):
                # (stack_size, H, W) -> (H, W, stack_size) for albumentations
                hwc = stacked[i].transpose(1, 2, 0)
                aug = self.augment_fn(image=hwc)["image"]  # (H, W, stack_size)
                augmented.append(aug.transpose(2, 0, 1))   # (stack_size, H, W)
            stacked = np.stack(augmented)                  # (N, stack_size, H, W)

        return torch.from_numpy(stacked), True

    def _build_labels(self, row: pd.Series) -> torch.Tensor:
        """Build the 12-label float tensor for one study.

        Synovitis label routing:
            - Gold rows (is_gold=1): use hard integer from 'Synovitis' column.
              Gold labels are authoritative; do not soften them.
            - Non-gold rows (is_gold=0): use 'synovitis_soft' float (0.22 or 0.63),
              providing calibrated probabilistic supervision for unaddressed silence.
              (0.63 if concurrent Effusion=1, else 0.22 -- from Phase 2 imputation)

        All other 11 labels:
            Hard integer from label_cols. NaN preserved for unaddressed silence.
            Never replaced with 0 -- NaN is masked in MaskedBCEWithLogitsLoss.
        """
        is_gold = bool(row.get("is_gold", 0))
        label_values = []

        for col in self.label_cols:
            if (
                col == SYNOVITIS_COL
                and not is_gold
                and SYNOVITIS_SOFT_COL in row.index
                and pd.notna(row.get(SYNOVITIS_SOFT_COL))
            ):
                # Non-gold Synovitis -> use soft float target
                label_values.append(float(row[SYNOVITIS_SOFT_COL]))
            else:
                raw = row.get(col, float("nan"))
                label_values.append(float(raw) if pd.notna(raw) else float("nan"))

        return torch.tensor(label_values, dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict:
        row = self.study_df.iloc[idx]
        study_uid = row["StudyInstanceUID"]
        study_dir = self.dicom_root / "train_series" / study_uid
        if not study_dir.exists():
            study_dir = self.dicom_root / study_uid

        try:
            series_list = load_study_series(str(study_dir))
        except Exception as e:
            warnings.warn(
                f"Failed to load study {study_uid}: {e}. Using zero tensors.",
                RuntimeWarning,
                stacklevel=2,
            )
            series_list = []

        # Bilateral check -- warning only, no filtering
        self._check_bilateral(series_list, study_uid)

        sagittal, sag_ok = self._load_plane_tensor(series_list, AnatomicalPlane.SAGITTAL)
        coronal,  cor_ok = self._load_plane_tensor(series_list, AnatomicalPlane.CORONAL)
        axial,    axl_ok = self._load_plane_tensor(series_list, AnatomicalPlane.AXIAL)

        plane_present = torch.tensor([sag_ok, cor_ok, axl_ok], dtype=torch.bool)
        labels = self._build_labels(row)

        return {
            "sagittal":      sagittal,
            "coronal":       coronal,
            "axial":         axial,
            "plane_present": plane_present,
            "labels":        labels,
            "is_gold":       int(row.get("is_gold", 0)),
            "study_uid":     study_uid,
        }
