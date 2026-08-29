"""
mri_dataset.py -- PyTorch Datasets for RSNA Knee MRI Studies

Two dataset classes:

1. KneeMRIDataset (legacy / fallback)
   Loads studies live from DICOM at __getitem__ time. Slow on Kaggle's
   network storage (~1.3s/study). Kept for compatibility and as a fallback
   when the preprocessing cache is not available.

2. KneeMRIDatasetCached (primary, fast)
   Loads studies from preprocessed .npz cache files written by
   scripts/preprocess_volumes.py (~5ms/study). Use this for training.
   Falls back to live DICOM load if a study's .npz is missing.

Key design decisions (shared by both classes):
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
PLANE_NAMES     = ["sagittal", "coronal", "axial"]
SERIES_PRIORITY = [SeriesType.FLUID_SENSITIVE, SeriesType.ANATOMICAL]

# Label column names for Synovitis routing
SYNOVITIS_COL      = "Synovitis"
SYNOVITIS_SOFT_COL = "synovitis_soft"


# ============================================================================
# Slice Selection
# ============================================================================

def _uid_to_seed(study_uid: str) -> int:
    """Derive a stable per-study integer seed from the StudyInstanceUID string.

    Ensures different studies sample different slices, but the same study
    always samples the same slices (reproducible). Replaces the broken
    fixed seed=42 (which caused identical slice selection for every study).

    The seed is taken modulo 2^31 to stay within numpy's integer range.
    """
    return abs(hash(study_uid)) % (2 ** 31)


def select_slices(
    volume: np.ndarray,
    n_slices: int,
    seed: int,
    center_weight: float = 2.0,
) -> np.ndarray:
    """Select N representative slices with center-weighted sampling per study.

    The central 50% of the volume (joint articular space) is sampled at
    2x density to over-represent the joint line -- where meniscal tears,
    cartilage loss, and effusion are most visible.

    Args:
        volume:        (S, H, W) float32 MRI volume, values in [0, 1].
        n_slices:      Target number of slices to select.
        seed:          Per-study RNG seed (use _uid_to_seed). NOT fixed to 42.
        center_weight: Density multiplier for central 50% of volume.

    Returns:
        (n_slices, H, W) float32 array. Zero-padded if volume < n_slices.
    """
    total = volume.shape[0]

    if total <= n_slices:
        pad = np.zeros((n_slices - total, *volume.shape[1:]), dtype=volume.dtype)
        return np.concatenate([volume, pad], axis=0)

    weights = np.ones(total, dtype=np.float32)
    weights[int(total * 0.25):int(total * 0.75)] = center_weight
    weights /= weights.sum()

    rng = np.random.default_rng(seed=seed)
    indices = np.sort(rng.choice(total, size=n_slices, replace=False, p=weights))
    return volume[indices]


# ============================================================================
# 2.5D Stacking (vectorized)
# ============================================================================

def stack_25d(volume: np.ndarray, stack_size: int = 3) -> np.ndarray:
    """Stack consecutive slices as 2.5D channels: (N, H, W) -> (N, stack_size, H, W).

    Each slice i gets channels: [slice i-half, ..., slice i, ..., slice i+half].
    Boundaries clamp to edge (no wrap-around). stack_size must be odd.

    Satisfies the clinical 2-slice meniscal tear rule: the model sees adjacent
    slices simultaneously in a single backbone forward pass, enabling detection
    of abnormalities spanning multiple consecutive MRI slices.

    Implementation uses vectorized numpy indexing (no Python loops) for speed.

    Args:
        volume:     (N, H, W) float32 slice volume.
        stack_size: Must be odd (e.g., 3 -> prev, current, next).

    Returns:
        (N, stack_size, H, W) float32 array.
    """
    assert stack_size % 2 == 1, f"stack_size must be odd, got {stack_size}"
    half = stack_size // 2
    n = volume.shape[0]

    # Build index array (n, stack_size) with edge clamping -- fully vectorized
    center_idx = np.arange(n)                               # (n,)
    offsets    = np.arange(-half, half + 1)                 # (stack_size,)
    idx = np.clip(
        center_idx[:, None] + offsets[None, :], 0, n - 1   # (n, stack_size)
    )
    return volume[idx]  # (n, stack_size, H, W)


# ============================================================================
# Shared Label Building
# ============================================================================

def _build_labels(row: pd.Series, label_cols: list[str]) -> torch.Tensor:
    """Build the 12-label float tensor for one study.

    Synovitis label routing:
        - Gold rows (is_gold=1): use hard integer from 'Synovitis' column.
        - Non-gold rows (is_gold=0): use 'synovitis_soft' float (0.22 or 0.63).

    All other 11 labels: hard integer. NaN preserved for unaddressed silence.
    """
    is_gold = bool(row.get("is_gold", 0))
    label_values = []

    for col in label_cols:
        if (
            col == SYNOVITIS_COL
            and not is_gold
            and SYNOVITIS_SOFT_COL in row.index
            and pd.notna(row.get(SYNOVITIS_SOFT_COL))
        ):
            label_values.append(float(row[SYNOVITIS_SOFT_COL]))
        else:
            raw = row.get(col, float("nan"))
            label_values.append(float(raw) if pd.notna(raw) else float("nan"))

    return torch.tensor(label_values, dtype=torch.float32)


# ============================================================================
# Augmentation Helper
# ============================================================================

def _augment_stacked(stacked: np.ndarray, augment_fn) -> np.ndarray:
    """Apply albumentations augmentation to all slices in a stacked volume.

    Args:
        stacked:    (N, stack_size, H, W) numpy array.
        augment_fn: Albumentations Compose transform.

    Returns:
        (N, stack_size, H, W) augmented array.
    """
    n = stacked.shape[0]
    augmented = []
    for i in range(n):
        # (stack_size, H, W) -> (H, W, stack_size) for albumentations
        hwc = stacked[i].transpose(1, 2, 0)
        aug = augment_fn(image=hwc)["image"]  # (H, W, stack_size)
        augmented.append(aug.transpose(2, 0, 1))  # (stack_size, H, W)
    return np.stack(augmented)  # (N, stack_size, H, W)


# ============================================================================
# KneeMRIDataset -- Legacy Live-DICOM Loader (kept for fallback)
# ============================================================================

class KneeMRIDataset(Dataset):
    """PyTorch Dataset for RSNA Knee MRI (live DICOM loading).

    Loads and preprocesses multi-series, multi-plane knee MRI studies at
    __getitem__ time. Correct but slow on Kaggle network storage (~1.3s/study).

    Use KneeMRIDatasetCached when the .npz preprocessing cache is available.

    Args:
        study_df:    DataFrame with StudyInstanceUID, label columns, is_gold.
        dicom_root:  DICOM root path.
        label_cols:  12 label column names from config.yaml labels.columns.
        n_slices:    Slices to select per plane series (default 24).
        target_size: Resize each slice to (H, W) (default (224, 224)).
        stack_size:  2.5D stacking depth -- must be odd (default 3).
        is_train:    Apply data augmentation if True.
        augment_fn:  Optional albumentations Compose transform. Applied AFTER
                     stack_25d so all channels share the same spatial transform.

    Returns per __getitem__:
        {
          "sagittal":      Tensor (N, stack_size, H, W) float32,
          "coronal":       Tensor (N, stack_size, H, W) float32,
          "axial":         Tensor (N, stack_size, H, W) float32,
          "plane_present": Tensor (3,) bool,
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
        self.study_df    = study_df.reset_index(drop=True)
        self.dicom_root  = Path(dicom_root)
        self.label_cols  = label_cols
        self.n_slices    = n_slices
        self.target_size = target_size
        self.stack_size  = stack_size
        self.is_train    = is_train
        self.augment_fn  = augment_fn

    def __len__(self) -> int:
        return len(self.study_df)

    def _resize_volume(self, volume: np.ndarray) -> np.ndarray:
        import cv2
        H, W = self.target_size
        return np.stack([
            cv2.resize(sl, (W, H), interpolation=cv2.INTER_LINEAR)
            for sl in volume
        ]).astype(np.float32)

    def _check_bilateral(self, series_list: list[DICOMSeries], study_uid: str) -> None:
        if len(series_list) > 12:
            warnings.warn(
                f"Study {study_uid} may be bilateral ({len(series_list)} series). "
                "Keeping all series (safe default -- no laterality filtering).",
                UserWarning,
                stacklevel=2,
            )

    def _select_best_series(
        self,
        series_list: list[DICOMSeries],
        plane: AnatomicalPlane,
    ) -> Optional[DICOMSeries]:
        candidates = [s for s in series_list if s.plane == plane]
        if not candidates:
            return None
        for stype in SERIES_PRIORITY:
            typed = [s for s in candidates if s.series_type == stype]
            if typed:
                return max(typed, key=lambda s: s.n_slices)
        return max(candidates, key=lambda s: s.n_slices)

    def _load_plane_tensor(
        self,
        series_list: list[DICOMSeries],
        plane: AnatomicalPlane,
        seed: int,
    ) -> tuple[torch.Tensor, bool]:
        series = self._select_best_series(series_list, plane)

        if series is None:
            return (
                torch.zeros(
                    self.n_slices, self.stack_size, *self.target_size,
                    dtype=torch.float32,
                ),
                False,
            )

        volume = series.pixel_array                               # (S, H, W) float32
        # FIX: select THEN resize -- avoids resizing discarded slices
        volume  = select_slices(volume, self.n_slices, seed)     # (N, H, W)
        volume  = self._resize_volume(volume)                     # (N, H, W) resized
        stacked = stack_25d(volume, self.stack_size)              # (N, stack_size, H, W)

        if self.is_train and self.augment_fn is not None:
            stacked = _augment_stacked(stacked, self.augment_fn)

        return torch.from_numpy(stacked), True

    def __getitem__(self, idx: int) -> dict:
        row = self.study_df.iloc[idx]
        study_uid = row["StudyInstanceUID"]
        seed = _uid_to_seed(study_uid)

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

        self._check_bilateral(series_list, study_uid)

        sagittal, sag_ok = self._load_plane_tensor(
            series_list, AnatomicalPlane.SAGITTAL, seed
        )
        coronal,  cor_ok = self._load_plane_tensor(
            series_list, AnatomicalPlane.CORONAL, seed
        )
        axial,    axl_ok = self._load_plane_tensor(
            series_list, AnatomicalPlane.AXIAL, seed
        )

        plane_present = torch.tensor([sag_ok, cor_ok, axl_ok], dtype=torch.bool)
        labels = _build_labels(row, self.label_cols)

        return {
            "sagittal":      sagittal,
            "coronal":       coronal,
            "axial":         axial,
            "plane_present": plane_present,
            "labels":        labels,
            "is_gold":       int(row.get("is_gold", 0)),
            "study_uid":     study_uid,
        }


# ============================================================================
# KneeMRIDatasetCached -- Fast Cache Loader (primary class for training)
# ============================================================================

class KneeMRIDatasetCached(Dataset):
    """PyTorch Dataset that loads from preprocessed .npz cache files.

    Each .npz file was written by scripts/preprocess_volumes.py and contains:
        sagittal, coronal, axial: (N, H, W) uint8 arrays
        plane_present:            (3,) bool array

    __getitem__ cost: ~5ms (np.load) vs ~1300ms (live DICOM decode).
    This enables full 5-fold 10-epoch training within a single Kaggle session.

    Falls back to live DICOM decode via KneeMRIDataset if a study's .npz
    is missing (e.g., preprocessing failed for that study). This ensures
    training always completes even with partial caches.

    Args:
        study_df:    DataFrame with StudyInstanceUID, label columns, is_gold.
        cache_dir:   Directory containing {StudyInstanceUID}.npz files.
        label_cols:  12 label column names from config.yaml labels.columns.
        n_slices:    Must match value used during preprocessing.
        target_size: Must match value used during preprocessing.
        stack_size:  2.5D stacking depth -- must be odd (default 3).
        is_train:    Apply data augmentation if True.
        augment_fn:  Optional albumentations Compose transform.
        dicom_root:  Optional fallback DICOM root for cache-miss studies.
    """

    def __init__(
        self,
        study_df: pd.DataFrame,
        cache_dir: str,
        label_cols: list[str],
        n_slices: int = 24,
        target_size: tuple[int, int] = (224, 224),
        stack_size: int = 3,
        is_train: bool = True,
        augment_fn=None,
        dicom_root: Optional[str] = None,
    ) -> None:
        self.study_df    = study_df.reset_index(drop=True)
        self.cache_dir   = Path(cache_dir)
        self.label_cols  = label_cols
        self.n_slices    = n_slices
        self.target_size = target_size
        self.stack_size  = stack_size
        self.is_train    = is_train
        self.augment_fn  = augment_fn
        self.dicom_root  = dicom_root

        # Count cache hits at init time for diagnostics
        n_cached = sum(
            1 for uid in self.study_df["StudyInstanceUID"]
            if (self.cache_dir / f"{uid}.npz").exists()
        )
        n_total = len(self.study_df)
        if n_cached < n_total:
            warnings.warn(
                f"KneeMRIDatasetCached: {n_cached}/{n_total} studies found in cache. "
                f"{n_total - n_cached} studies will use slow live DICOM fallback. "
                "Run preprocess_volumes.preprocess_dataset() to populate the cache.",
                UserWarning,
                stacklevel=2,
            )

    def __len__(self) -> int:
        return len(self.study_df)

    def _load_from_cache(self, study_uid: str) -> Optional[dict]:
        """Load preprocessed arrays from .npz cache. Returns None on miss/error."""
        cache_path = self.cache_dir / f"{study_uid}.npz"
        if not cache_path.exists():
            return None
        try:
            data = np.load(str(cache_path))
            return {
                "sagittal":      data["sagittal"].astype(np.float32) / 255.0,
                "coronal":       data["coronal"].astype(np.float32) / 255.0,
                "axial":         data["axial"].astype(np.float32) / 255.0,
                "plane_present": data["plane_present"],
            }
        except Exception as e:
            warnings.warn(
                f"Failed to load cache for {study_uid}: {e}. Will use DICOM fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    def _volume_to_tensor(
        self,
        volume: np.ndarray,
        plane_ok: bool,
        seed: int,
    ) -> tuple[torch.Tensor, bool]:
        """Convert (N, H, W) float32 volume to (N, stack_size, H, W) tensor.

        Volume from cache is already selected (N slices) and resized (H, W).
        We only apply stack_25d and optional augmentation here.

        Note: For train-time diversity, we could re-sample from a 2N-slice
        cache here. For simplicity, the current implementation uses the fixed
        N slices stored in cache. This is consistent and reproducible.
        """
        if not plane_ok:
            return (
                torch.zeros(
                    self.n_slices, self.stack_size, *self.target_size,
                    dtype=torch.float32,
                ),
                False,
            )

        stacked = stack_25d(volume, self.stack_size)  # (N, stack_size, H, W)

        if self.is_train and self.augment_fn is not None:
            stacked = _augment_stacked(stacked, self.augment_fn)

        return torch.from_numpy(stacked), True

    def __getitem__(self, idx: int) -> dict:
        row = self.study_df.iloc[idx]
        study_uid = row["StudyInstanceUID"]
        seed = _uid_to_seed(study_uid)

        cache_data = self._load_from_cache(study_uid)

        if cache_data is not None:
            # Fast path: cache hit (~5ms)
            plane_present_arr = cache_data["plane_present"]
            sagittal, sag_ok = self._volume_to_tensor(
                cache_data["sagittal"], bool(plane_present_arr[0]), seed
            )
            coronal,  cor_ok = self._volume_to_tensor(
                cache_data["coronal"],  bool(plane_present_arr[1]), seed
            )
            axial,    axl_ok = self._volume_to_tensor(
                cache_data["axial"],    bool(plane_present_arr[2]), seed
            )
        else:
            # Slow fallback: live DICOM decode (~1300ms)
            if self.dicom_root is None:
                warnings.warn(
                    f"Cache miss for {study_uid} and no dicom_root provided. "
                    "Returning zero tensors.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                sagittal = coronal = axial = torch.zeros(
                    self.n_slices, self.stack_size, *self.target_size,
                    dtype=torch.float32,
                )
                sag_ok = cor_ok = axl_ok = False
            else:
                fallback_ds = KneeMRIDataset(
                    study_df=self.study_df.iloc[[idx]].reset_index(drop=True),
                    dicom_root=self.dicom_root,
                    label_cols=self.label_cols,
                    n_slices=self.n_slices,
                    target_size=self.target_size,
                    stack_size=self.stack_size,
                    is_train=self.is_train,
                    augment_fn=self.augment_fn,
                )
                fb = fallback_ds[0]
                sagittal = fb["sagittal"]
                coronal  = fb["coronal"]
                axial    = fb["axial"]
                sag_ok   = fb["plane_present"][0].item()
                cor_ok   = fb["plane_present"][1].item()
                axl_ok   = fb["plane_present"][2].item()

        plane_present = torch.tensor([sag_ok, cor_ok, axl_ok], dtype=torch.bool)
        labels = _build_labels(row, self.label_cols)

        return {
            "sagittal":      sagittal,
            "coronal":       coronal,
            "axial":         axial,
            "plane_present": plane_present,
            "labels":        labels,
            "is_gold":       int(row.get("is_gold", 0)),
            "study_uid":     study_uid,
        }
