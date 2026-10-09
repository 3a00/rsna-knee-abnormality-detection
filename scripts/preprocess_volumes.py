"""scripts/preprocess_volumes.py

One-Time 5-Slice DICOM Preprocessing Cache for RSNA Efficiency Track Training.
Pre-extracts 5 centered physical slices per present slot into session scratch.
Features atomic file writes, pool initializer thread hygiene, and manifest verification.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import pandas as pd
import torch

from src.datasets.efficiency_pipeline import (
    DEFAULT_CROP_MM,
    load_series_metadata,
    process_study_to_5slice_cache,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_worker_series_metadata: dict | None = None


def _init_pool_worker(series_metadata: dict | None) -> None:
    """Initialize worker process: bind metadata and clamp intra-op threads."""
    global _worker_series_metadata
    _worker_series_metadata = series_metadata
    torch.set_num_threads(1)
    cv2.setNumThreads(0)


def _process_single_study(
    study_uid: str,
    dicom_root: Path,
    cache_dir: Path,
    target_size: int,
    crop_mm: float,
) -> tuple[bool, bool]:
    """Worker task processing and atomically persisting a single study cache.

    Returns:
        (success: bool, is_empty: bool)
    """
    out_file = cache_dir / f"{study_uid}.pt"
    if out_file.is_file():
        return True, False

    study_dir = dicom_root / study_uid
    if not study_dir.is_dir():
        candidates = list(dicom_root.glob(f"*/{study_uid}"))
        if candidates and candidates[0].is_dir():
            study_dir = candidates[0]
        else:
            return False, False

    try:
        tensor, mask = process_study_to_5slice_cache(
            study_dir=study_dir,
            series_metadata=_worker_series_metadata,
            num_workers=0,
            target_size=target_size,
            crop_mm=crop_mm,
            slice_pos=0.5,
        )
        is_empty = bool(mask.sum().item() == 0)
        payload = {
            "image": tensor,  # (6, 5, 336, 336) float16
            "presence_mask": mask,  # (6,) float32
            "meta": {
                "target_size": target_size,
                "crop_mm": float(crop_mm),
                "n_slices_cached": 5,
            },
        }
        tmp_file = out_file.with_suffix(".pt.tmp")
        torch.save(payload, tmp_file)
        os.replace(tmp_file, out_file)
        return True, is_empty
    except Exception as e:
        logger.warning("Failed caching study %s: %s", study_uid, e)
        return False, False


def preprocess_dataset(
    dicom_root: str | Path,
    cache_dir: str | Path,
    series_csv: str | Path | None = None,
    splits_csv: str | Path | None = None,
    labels_csv: str | Path | None = None,
    target_size: int = 336,
    crop_mm: float = DEFAULT_CROP_MM,
    n_workers: int = 4,
) -> None:
    """Pre-caches all training and validation studies into session scratch."""
    dicom_root = Path(dicom_root)
    cache_path = Path(cache_dir)

    try:
        cache_path.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        logger.error("Cannot create or write to cache directory %s: %s", cache_path, e)
        raise RuntimeError(f"Cache directory {cache_path} is not writable. Aborting.") from e

    series_metadata = load_series_metadata(series_csv) if series_csv else None

    # Discover StudyInstanceUIDs
    study_uids: list[str] = []
    if splits_csv and Path(splits_csv).is_file():
        df = pd.read_csv(splits_csv)
        study_uids = df["StudyInstanceUID"].astype(str).unique().tolist()
    elif labels_csv and Path(labels_csv).is_file():
        df = pd.read_csv(labels_csv)
        study_uids = df["StudyInstanceUID"].astype(str).unique().tolist()
    else:
        study_uids = [p.name for p in dicom_root.iterdir() if p.is_dir() and not p.name.startswith(".")]

    logger.info("Found %d studies to pre-cache in %s", len(study_uids), cache_path)
    pending_uids = [u for u in study_uids if not (cache_path / f"{u}.pt").is_file()]
    logger.info("Pending studies to decode: %d (already cached: %d)", len(pending_uids), len(study_uids) - len(pending_uids))

    # Pre-flight disk space check: compute required bytes (6.8 MB per study)
    required_bytes = int(len(pending_uids) * 6.8 * 1024 * 1024)
    free_bytes = shutil.disk_usage(cache_path).free
    logger.info("Required disk space: %.2f GB (Available: %.2f GB)", required_bytes / 1e9, free_bytes / 1e9)
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Insufficient disk space in {cache_path}: requires {required_bytes / 1e9:.1f} GB, "
            f"but only {free_bytes / 1e9:.1f} GB is free."
        )

    # Provenance guard on resume: check existing manifest before continuing
    manifest_file = cache_path / "cache_manifest.json"
    existing_pt_files = list(cache_path.glob("*.pt"))
    old_manifest: dict[str, Any] = {}

    if manifest_file.is_file():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                old_manifest = json.load(f)
            if (
                int(old_manifest.get("target_size", target_size)) != target_size
                or abs(float(old_manifest.get("crop_mm", crop_mm)) - crop_mm) > 1e-4
                or int(old_manifest.get("n_slices_cached", 5)) != 5
            ):
                raise RuntimeError(
                    f"{cache_path} holds a cache with different provenance {old_manifest}; "
                    f"requested size={target_size}, crop={crop_mm}, slices=5. Please use a fresh directory."
                )
        except Exception as e:
            if "provenance" in str(e):
                raise
            logger.warning("Could not parse existing manifest: %s", e)
    else:
        if existing_pt_files:
            raise RuntimeError(
                f"Found {len(existing_pt_files)} cached .pt files in {cache_path} without cache_manifest.json. "
                f"Cannot verify provenance of existing files. Clean directory or remove files to proceed."
            )
        # Write provenance-first in-progress manifest before processing begins
        init_manifest = {
            "target_size": target_size,
            "crop_mm": float(crop_mm),
            "n_slices_cached": 5,
            "total_studies_expected": len(study_uids),
            "total_cached_files": 0,
            "failed_uids": [],
            "empty_uids": [],
            "in_progress": True,
            "created_at": time.time(),
        }
        manifest_file.write_text(json.dumps(init_manifest, indent=2))
        logger.info("Initialized in-progress cache manifest at %s", manifest_file)

    t0 = time.time()
    this_run_failed: list[str] = []
    this_run_empty: list[str] = []
    success_cnt = 0

    if pending_uids:
        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_pool_worker, initargs=(series_metadata,)) as executor:
                futures = {
                    executor.submit(_process_single_study, uid, dicom_root, cache_path, target_size, crop_mm): uid
                    for uid in pending_uids
                }
                for idx, fut in enumerate(futures):
                    uid = futures[fut]
                    ok, is_empty = fut.result()
                    if ok:
                        success_cnt += 1
                        if is_empty:
                            this_run_empty.append(uid)
                    else:
                        this_run_failed.append(uid)
                    if (idx + 1) % 100 == 0 or (idx + 1) == len(pending_uids):
                        elapsed = time.time() - t0
                        logger.info("Cached [%d/%d] studies (%.1f%%, %.1fs)", idx + 1, len(pending_uids), 100.0 * (idx + 1) / len(pending_uids), elapsed)
        else:
            _init_pool_worker(series_metadata)
            for idx, uid in enumerate(pending_uids):
                ok, is_empty = _process_single_study(uid, dicom_root, cache_path, target_size, crop_mm)
                if ok:
                    success_cnt += 1
                    if is_empty:
                        this_run_empty.append(uid)
                else:
                    this_run_failed.append(uid)
                if (idx + 1) % 100 == 0:
                    logger.info("Cached [%d/%d] studies", idx + 1, len(pending_uids))

        elapsed = time.time() - t0
        logger.info("Pre-caching finished: %d succeeded, %d failed, %d empty in %.1fs", success_cnt, len(this_run_failed), len(this_run_empty), elapsed)

    # Accumulate failed and empty UIDs across runs
    accum_failed = {
        u for u in set(old_manifest.get("failed_uids", [])) | set(this_run_failed)
        if not (cache_path / f"{u}.pt").is_file()
    }
    accum_empty = set(old_manifest.get("empty_uids", [])) | set(this_run_empty)

    manifest_data = {
        "target_size": target_size,
        "crop_mm": float(crop_mm),
        "n_slices_cached": 5,
        "total_studies_expected": len(study_uids),
        "total_cached_files": len(list(cache_path.glob("*.pt"))),
        "failed_uids": sorted(accum_failed),
        "empty_uids": sorted(accum_empty),
        "in_progress": False,
        "updated_at": time.time(),
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info("Wrote cache manifest to %s", manifest_file)

    if len(accum_failed) > max(1, int(0.005 * len(study_uids))):
        logger.error("Pre-caching failure rate exceeded 0.5%% (%d failures): %s", len(accum_failed), sorted(accum_failed)[:5])
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-cache 5-slice MRI volumes")
    parser.add_argument("--dicom-root", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="/kaggle/tmp/rsna_cache")
    parser.add_argument("--series-csv", type=str, default=None)
    parser.add_argument("--splits-csv", type=str, default=None)
    parser.add_argument("--labels-csv", type=str, default=None)
    parser.add_argument("--target-size", type=int, default=336)
    parser.add_argument("--crop-mm", type=float, default=DEFAULT_CROP_MM)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    preprocess_dataset(
        dicom_root=args.dicom_root,
        cache_dir=args.cache_dir,
        series_csv=args.series_csv,
        splits_csv=args.splits_csv,
        labels_csv=args.labels_csv,
        target_size=args.target_size,
        crop_mm=args.crop_mm,
        n_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
