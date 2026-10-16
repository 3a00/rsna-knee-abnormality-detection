"""scripts/preprocess_336px_cache.py

Dedicated Offline 336px Pre-Caching Pipeline for RSNA Efficiency Track Training.
Pre-extracts 5 centered physical slices per present slot, crops 140mm physical FOV,
and persists compressed .npz archives per study with atomic file replacement.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch

from src.datasets.efficiency_pipeline import (
    DEFAULT_CROP_MM,
    SLOT_NAMES,
    load_series_metadata,
    process_study_to_5slice_cache,
)
from src.training.train_efficiency import seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_worker_series_metadata: dict | None = None


def _init_pool_worker(series_metadata: dict | None) -> None:
    """Initialize worker process: bind shared series metadata and disable OpenCV multithreading."""
    global _worker_series_metadata
    _worker_series_metadata = series_metadata
    torch.set_num_threads(1)
    cv2.setNumThreads(0)


def _process_single_study_npz(
    study_uid: str,
    dicom_root: Path,
    cache_dir: Path,
    target_size: int,
    crop_mm: float,
) -> tuple[bool, bool, bool, list[int], str]:
    """Process single study and save as compressed .npz using a direct binary file handle.

    Writing to an open binary file object prevents NumPy from appending duplicate '.npz' extensions.

    Returns:
        tuple of (success: bool, is_empty: bool, is_clipped: bool, active_slots: list[int], uid: str)
    """
    out_file = cache_dir / f"{study_uid}.npz"
    if out_file.is_file():
        try:
            with open(out_file, "rb") as fp:
                data = np.load(fp)
                mask = data["presence_mask"]
                active = [i for i, m in enumerate(mask) if m > 0]
                is_clipped = bool(data["clipped"]) if "clipped" in data.files else False
                return True, len(active) == 0, is_clipped, active, study_uid
        except Exception:
            pass

    study_dir = dicom_root / study_uid
    if not study_dir.is_dir():
        candidates = list(dicom_root.glob(f"*/{study_uid}"))
        if candidates and candidates[0].is_dir():
            study_dir = candidates[0]
        else:
            return False, False, False, [], study_uid

    try:
        res = process_study_to_5slice_cache(
            study_dir=study_dir,
            series_metadata=_worker_series_metadata,
            num_workers=0,
            target_size=target_size,
            crop_mm=crop_mm,
            slice_pos=0.5,
            return_info=True,
        )
        tensor, mask, info = res  # type: ignore
        active_slots = info.get("active_slots", [i for i, m in enumerate(mask.tolist()) if m > 0])
        is_empty = len(active_slots) == 0
        is_clipped = bool(info.get("clipped", False))

        # yagni: write via binary file handle so numpy never appends '.npz' to temp name
        tmp_file = out_file.with_name(out_file.name + ".tmp")
        try:
            with open(tmp_file, "wb") as f:
                np.savez_compressed(
                    f,
                    image=tensor.cpu().numpy().astype(np.float16),
                    presence_mask=mask.cpu().numpy().astype(np.float32),
                    clipped=np.uint8(is_clipped),
                )
            os.replace(tmp_file, out_file)
        finally:
            tmp_file.unlink(missing_ok=True)

        return True, is_empty, is_clipped, active_slots, study_uid
    except Exception as e:
        logger.warning("Failed caching study %s: %s", study_uid, e)
        return False, False, False, [], study_uid


def export_qc_montage(cache_dir: Path, out_path: Path, n_samples: int = 16) -> None:
    """Generate visual QC montage of center slices at native cached scale for 16-24 studies."""
    npz_files = sorted(cache_dir.glob("*.npz"))[:n_samples]
    if not npz_files:
        logger.warning("No .npz files found in %s to generate QC montage", cache_dir)
        return

    montage_rows = []
    for f in npz_files:
        with open(f, "rb") as fp:
            data = np.load(fp)
            img = data["image"]  # (6, 5, H, W)
            mask = data["presence_mask"]

        # Center slice index 2 for all 6 slots at native normalized scale [-2.5, 2.5] -> [0, 255]
        row_slots = []
        for s in range(6):
            if mask[s] > 0:
                sl = img[s, 2].astype(np.float32)
                sl_disp = np.clip((sl + 2.5) / 5.0 * 255.0, 0, 255).astype(np.uint8)
            else:
                sl_disp = np.zeros((img.shape[2], img.shape[3]), dtype=np.uint8)
            row_slots.append(sl_disp)
        montage_rows.append(np.hstack(row_slots))

    full_montage = np.vstack(montage_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), full_montage)
    logger.info("Exported %d-study QC montage to %s", len(npz_files), out_path)


def benchmark_loader_throughput(cache_dir: Path, study_uids: list[str], n_trials: int = 50) -> float:
    """Benchmark raw DataLoader read + float cast + augmentation throughput per study."""
    from src.training.train_efficiency import apply_msk_augmentations

    torch.set_num_threads(1)
    cv2.setNumThreads(0)

    sample_uids = study_uids[:min(n_trials, len(study_uids))]
    t0 = time.time()
    for uid in sample_uids:
        f = cache_dir / f"{uid}.npz"
        if not f.is_file():
            continue
        with open(f, "rb") as fp:
            data = np.load(fp)
            img = torch.from_numpy(data["image"])
            mask = torch.from_numpy(data["presence_mask"]).float()
        # Slice first, then convert to float
        img = img[:, 1:4].contiguous().float()
        img = apply_msk_augmentations(img, mask)

    elapsed = time.time() - t0
    rate = len(sample_uids) / max(1e-4, elapsed)
    logger.info("Loader benchmark: %.1f studies/s (%.3f s/study) [Note: files in page cache]", rate, elapsed / max(1, len(sample_uids)))
    return rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess and cache 336px 5-slice MRI studies into compressed .npz")
    parser.add_argument("--dicom-root", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="/kaggle/tmp/rsna_cache_336_140mm")
    parser.add_argument("--series-csv", type=str, default=None)
    parser.add_argument("--splits-csv", type=str, required=True)
    parser.add_argument("--target-size", type=int, default=336)
    parser.add_argument("--crop-mm", type=float, default=DEFAULT_CROP_MM)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Process random pilot subset (e.g. 50)")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--qc-montage", type=str, default=None, help="Path for QC visual montage image (16-24 studies)")
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--max-fail-frac", type=float, default=0.005, help="Allowed failure fraction (default: 0.5%%)")
    args = parser.parse_args()
    seed_everything(42)

    # Route pilot run into dedicated directory to prevent poisoning full cache
    if args.limit and args.cache_dir == "/kaggle/tmp/rsna_cache_336_140mm":
        args.cache_dir = "/kaggle/tmp/rsna_cache_pilot"
        logger.info("Pilot mode: routing cache to %s", args.cache_dir)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dicom_root = Path(args.dicom_root)

    # 1a. If --overwrite is set, delete all existing .npz and .tmp files first
    if args.overwrite and cache_dir.exists():
        logger.info("--overwrite specified: purging existing cache files in %s", cache_dir)
        for p in (*cache_dir.glob("*.npz"), *cache_dir.glob("*.tmp")):
            p.unlink(missing_ok=True)

    manifest_p = cache_dir / "cache_manifest.json"
    if manifest_p.is_file() and not args.overwrite:
        with open(manifest_p, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if (
            existing.get("target_size") != args.target_size
            or abs(float(existing.get("crop_mm", 0.0)) - args.crop_mm) > 1e-4
            or existing.get("format") != "npz"
        ):
            raise RuntimeError(
                f"Conflicting cache found at {cache_dir}: {existing}. "
                "Specify a clean directory or pass --overwrite."
            )

    splits_df = pd.read_csv(args.splits_csv)
    all_uids = splits_df["StudyInstanceUID"].astype(str).unique().tolist()

    if args.limit:
        rng = np.random.default_rng(42)
        study_uids = sorted(rng.choice(all_uids, size=min(args.limit, len(all_uids)), replace=False).tolist())
    else:
        study_uids = all_uids

    logger.info("Processing %d studies at %d px (crop %.1f mm)", len(study_uids), args.target_size, args.crop_mm)
    series_meta = load_series_metadata(args.series_csv) if args.series_csv else None

    manifest_data = {
        "target_size": args.target_size,
        "crop_mm": float(args.crop_mm),
        "n_slices_cached": 5,
        "format": "npz",
        "in_progress": True,
        "start_time": time.time(),
    }
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # Do not carry old failed_uids over: all missing studies are re-attempted cleanly
    failed_uids: list[str] = []
    empty_uids: list[str] = []
    clipped_uids: list[str] = []
    slot_counts = [0] * 6
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.num_workers, initializer=_init_pool_worker, initargs=(series_meta,)) as ex:
        futures = {
            ex.submit(_process_single_study_npz, uid, dicom_root, cache_dir, args.target_size, args.crop_mm): uid
            for uid in study_uids
        }
        for i, fut in enumerate(as_completed(futures)):
            uid = futures[fut]
            try:
                success, is_empty, is_clipped, active_slots, _ = fut.result()
                if not success:
                    failed_uids.append(uid)
                else:
                    if is_empty:
                        empty_uids.append(uid)
                    if is_clipped:
                        clipped_uids.append(uid)
                    for s in active_slots:
                        slot_counts[s] += 1
            except Exception as err:
                logger.error("Study %s threw worker exception: %s", uid, err)
                failed_uids.append(uid)

            if (i + 1) % 50 == 0 or (i + 1) == len(study_uids):
                elapsed = time.time() - t0
                rate = (i + 1) / max(0.1, elapsed)
                logger.info(
                    "Cached %d/%d (%.1f studies/s, %d failed, %d empty, %d clipped)",
                    i + 1, len(study_uids), rate, len(failed_uids), len(empty_uids), len(clipped_uids)
                )

    total_time = time.time() - t0
    manifest_data["in_progress"] = False
    manifest_data["elapsed_sec"] = total_time
    manifest_data["total_studies"] = len(study_uids)
    manifest_data["failed_uids"] = failed_uids
    manifest_data["empty_uids"] = empty_uids
    manifest_data["clipped_uids"] = clipped_uids
    manifest_data["slot_presence_rates"] = {
        SLOT_NAMES[i]: slot_counts[i] / max(1, len(study_uids)) for i in range(6)
    }

    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    sample_files = list(cache_dir.glob("*.npz"))[:50]
    avg_mb = (sum(f.stat().st_size for f in sample_files) / max(1, len(sample_files))) / (1024 * 1024) if sample_files else 0.0
    n_full = len(all_uids)
    free_disk_gb = shutil.disk_usage(cache_dir).free / (1024**3)

    logger.info("==================================================")
    logger.info("PRE-CACHING REPORT:")
    logger.info("  Processed: %d studies in %.2f min (%.2f s/study)", len(study_uids), total_time / 60.0, total_time / max(1, len(study_uids)))
    logger.info("  Average File Size: %.2f MB/study", avg_mb)
    logger.info("  Projected full cache (%d studies): %.0f min, %.1f GB | free disk: %.1f GB",
                n_full, (total_time / max(1, len(study_uids)) * n_full) / 60.0,
                avg_mb * n_full / 1024.0, free_disk_gb)
    logger.info("  Slot Presence Rates: %s", {k: f"{v:.1%}" for k, v in manifest_data["slot_presence_rates"].items()})
    logger.info("  Empty Studies: %d | Clipped Studies: %d | Failures: %d", len(empty_uids), len(clipped_uids), len(failed_uids))
    logger.info("==================================================")

    if args.qc_montage:
        export_qc_montage(cache_dir, Path(args.qc_montage), n_samples=min(20, len(study_uids)))

    if args.benchmark:
        benchmark_loader_throughput(cache_dir, study_uids, n_trials=50)

    max_allowed_failures = max(1, int(args.max_fail_frac * len(study_uids)))
    if len(failed_uids) > max_allowed_failures:
        logger.error(
            "Failures (%d) exceed allowed threshold (%d). Exiting 1.",
            len(failed_uids), max_allowed_failures
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
