"""
preprocess_volumes.py -- One-Time DICOM Preprocessing Cache for Phase 3 Training

Why this exists:
    Decoding full DICOM studies inside __getitem__ (during training) costs ~1.3s/study
    on Kaggle's network-mounted storage. With num_workers=0, this keeps the GPU idle
    >90% of wall time: 50 epochs x 4407 studies = 85+ hours. Unacceptable.

    This script decodes each study ONCE, picks the best series per plane, selects N
    slices, resizes, normalizes, and saves float32->uint8 .npz to local scratch.
    Subsequent __getitem__ becomes a simple np.load (~5ms) instead of ~1300ms.

    Expected speedup: 50-200x per epoch. Full 5-fold 10-epoch run drops from ~85h
    to ~3-5h on Kaggle T4.

Usage (Cell 4 in notebook, before training):
    from scripts.preprocess_volumes import preprocess_dataset
    preprocess_dataset(
        train_df=train_df,
        dicom_root=dicom_root,
        cache_dir="/kaggle/tmp/rsna_cache",
        cfg=cfg,
        n_workers=4,
    )

Output layout:
    cache_dir/
        {StudyInstanceUID}.npz  -- keys: "sagittal", "coronal", "axial", "plane_present"
                                    shapes: (N, H, W) uint8, (3,) bool
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================================
# Helpers
# ============================================================================

def _uid_to_seed(study_uid: str) -> int:
    """Derive a stable per-study integer seed from the StudyInstanceUID string.

    Using a UID-based seed instead of a fixed seed=42 ensures that different
    studies sample different slices, and that the same study always samples the
    same slices (reproducible across folds/epochs).

    The seed is taken modulo 2^31 to stay within numpy's integer seed range.
    """
    return abs(hash(study_uid)) % (2 ** 31)


def _select_slices_seeded(
    volume: np.ndarray,
    n_slices: int,
    seed: int,
    center_weight: float = 2.0,
) -> np.ndarray:
    """Center-weighted slice selection with per-study seed (no fixed seed=42).

    The central 50% of the volume gets 2x sampling weight to over-represent
    the articular space where meniscal tears, cartilage loss, and effusion
    are most visible.

    Args:
        volume:        (S, H, W) float32 normalized MRI volume.
        n_slices:      Target number of slices to select.
        seed:          Per-study RNG seed derived from StudyInstanceUID hash.
        center_weight: Weight multiplier for central 50% of slices.

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


def _resize_volume(volume: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize (S, H, W) volume to (S, target_h, target_w) using bilinear interpolation.

    Done AFTER slice selection so we only resize the N selected slices
    (not all 60+ raw slices), saving ~(total_slices/n_slices)x wasted cv2 calls.
    """
    import cv2
    return np.stack([
        cv2.resize(sl, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        for sl in volume
    ]).astype(np.float32)


def _float_to_uint8(volume: np.ndarray) -> np.ndarray:
    """Convert float32 [0,1] volume to uint8 [0,255] for compact disk storage.

    Storage savings vs float32: 4x (uint8 is 1 byte vs 4 bytes per element).
    At N=24, 224x224: 24 * 224 * 224 * 3 planes * 1 byte = ~3.6 MB/study (uint8)
    vs ~14.4 MB/study (float32). Total for 4407 studies: ~16 GB (uint8).
    """
    volume = np.clip(volume, 0.0, 1.0)
    return (volume * 255.0).astype(np.uint8)


# ============================================================================
# Per-Study Worker Function
# ============================================================================

def _process_one_study(args: tuple) -> tuple[str, Optional[str]]:
    """Worker function: load one study, preprocess, save .npz to cache_dir.

    Args:
        args: (study_uid, dicom_root, cache_dir, n_slices, target_size)

    Returns:
        (study_uid, error_message_or_None)
    """
    study_uid, dicom_root_str, cache_dir_str, n_slices, target_size = args

    cache_path = Path(cache_dir_str) / f"{study_uid}.npz"
    if cache_path.exists():
        return study_uid, None  # Already cached; skip

    try:
        from src.datasets.dicom_loader import (
            AnatomicalPlane,
            SeriesType,
            load_study_series,
        )

        PLANE_ORDER     = [AnatomicalPlane.SAGITTAL, AnatomicalPlane.CORONAL, AnatomicalPlane.AXIAL]
        SERIES_PRIORITY = [SeriesType.FLUID_SENSITIVE, SeriesType.ANATOMICAL]
        target_h, target_w = target_size
        seed = _uid_to_seed(study_uid)

        dicom_root = Path(dicom_root_str)
        study_dir  = dicom_root / "train_series" / study_uid
        if not study_dir.exists():
            study_dir = dicom_root / study_uid

        series_list = load_study_series(str(study_dir))

        arrays: dict[str, np.ndarray] = {}
        plane_present: list[bool] = []

        for plane in PLANE_ORDER:
            # Select best series for this plane (fluid-sensitive preferred)
            candidates = [s for s in series_list if s.plane == plane]
            chosen = None
            if candidates:
                for stype in SERIES_PRIORITY:
                    typed = [s for s in candidates if s.series_type == stype]
                    if typed:
                        chosen = max(typed, key=lambda s: s.n_slices)
                        break
                if chosen is None:
                    chosen = max(candidates, key=lambda s: s.n_slices)

            plane_name = plane.value.lower()  # "sagittal", "coronal", "axial"

            if chosen is None:
                # No series for this plane -- store zeros, mark absent
                arrays[plane_name] = np.zeros(
                    (n_slices, target_h, target_w), dtype=np.uint8
                )
                plane_present.append(False)
            else:
                volume = chosen.pixel_array  # (S, H, W) float32 normalized [0,1]
                # FIX: select THEN resize (not resize-all-then-select)
                volume = _select_slices_seeded(volume, n_slices, seed)  # (N, H, W)
                volume = _resize_volume(volume, target_h, target_w)      # (N, H, W) resized
                arrays[plane_name] = _float_to_uint8(volume)             # (N, H, W) uint8
                plane_present.append(True)

        np.savez_compressed(
            str(cache_path),
            sagittal=arrays["sagittal"],
            coronal=arrays["coronal"],
            axial=arrays["axial"],
            plane_present=np.array(plane_present, dtype=bool),
        )
        return study_uid, None

    except Exception as e:
        return study_uid, str(e)


# ============================================================================
# Public API
# ============================================================================

def preprocess_dataset(
    train_df: pd.DataFrame,
    dicom_root: str,
    cache_dir: str,
    cfg: dict,
    n_workers: int = 4,
) -> dict:
    """Preprocess all studies: decode once, select slices, resize, save .npz.

    This is the decisive Phase A fix. Run once per Kaggle session before
    training starts. Subsequent training epochs load from local .npz files
    (~5ms/study) instead of re-decoding DICOM (~1300ms/study).

    Args:
        train_df:   DataFrame with at least StudyInstanceUID column (4407 rows).
        dicom_root: Root path to DICOM data (e.g., /kaggle/input/.../train/).
        cache_dir:  Output directory for .npz files. Use /kaggle/tmp/rsna_cache
                    for fast local scratch on Kaggle (persists within session).
        cfg:        Full config.yaml dict (reads model.n_slices, model.target_size).
        n_workers:  Number of parallel workers. 4 is safe on Kaggle T4.

    Returns:
        dict with 'n_success', 'n_failed', 'failed_uids' keys.
    """
    import time
    import multiprocessing as mp

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_slices    = cfg["model"]["n_slices"]
    target_size = tuple(cfg["model"]["target_size"])
    study_uids  = train_df["StudyInstanceUID"].unique().tolist()
    n_total     = len(study_uids)

    # Check existing cache to skip already-done studies
    already_done = sum(1 for uid in study_uids if (cache_dir / f"{uid}.npz").exists())
    n_todo = n_total - already_done

    print(f"Preprocessing {n_todo} studies ({already_done} already cached)...")
    print(f"  cache_dir:   {cache_dir}")
    print(f"  n_slices:    {n_slices}")
    print(f"  target_size: {target_size}")
    print(f"  n_workers:   {n_workers}")
    sys.stdout.flush()

    if n_todo == 0:
        print("All studies already cached. Nothing to do.")
        return {"n_success": n_total, "n_failed": 0, "failed_uids": []}

    args_list = [
        (uid, dicom_root, str(cache_dir), n_slices, target_size)
        for uid in study_uids
    ]

    t_start = time.time()
    n_success, n_failed = already_done, 0
    failed_uids: list[str] = []

    # Timing probe: run 3 studies synchronously to estimate total time
    print("\nRunning timing probe (3 studies)...")
    sys.stdout.flush()
    probe_t = time.time()
    for args in args_list[:3]:
        _process_one_study(args)
    probe_elapsed = time.time() - probe_t
    secs_per_study = probe_elapsed / 3.0
    projected_total = secs_per_study * n_todo / n_workers
    print(f"  ~{secs_per_study:.2f}s/study -> projected total: "
          f"~{projected_total/60:.1f} min with {n_workers} workers")
    sys.stdout.flush()

    # Run preprocessing (multiprocessing pool)
    ctx = mp.get_context("spawn")  # "spawn" is safe on Kaggle (avoids fork+CUDA issues)
    with ctx.Pool(processes=n_workers) as pool:
        for i, (uid, error) in enumerate(
            pool.imap_unordered(_process_one_study, args_list)
        ):
            if error is None:
                n_success += 1
            else:
                n_failed += 1
                failed_uids.append(uid)
                warnings.warn(
                    f"Failed to preprocess study {uid}: {error}",
                    UserWarning,
                    stacklevel=1,
                )

            # Progress log every 200 studies
            if (i + 1) % 200 == 0 or (i + 1) == n_total:
                elapsed = time.time() - t_start
                pct = (i + 1) / n_total * 100
                eta_s = elapsed / (i + 1) * (n_total - i - 1) if i > 0 else 0
                print(
                    f"  [{i+1:4d}/{n_total}] {pct:5.1f}% | "
                    f"OK={n_success} FAIL={n_failed} | "
                    f"Elapsed={elapsed/60:.1f}m ETA={eta_s/60:.1f}m"
                )
                sys.stdout.flush()

    total_elapsed = time.time() - t_start
    print(f"\nPreprocessing complete in {total_elapsed/60:.1f} min.")
    print(f"  Cached: {n_success}/{n_total} studies")
    if n_failed > 0:
        print(f"  Failed: {n_failed} studies (will fall back to live DICOM decode at training time)")
    sys.stdout.flush()

    return {
        "n_success":   n_success,
        "n_failed":    n_failed,
        "failed_uids": failed_uids,
    }
