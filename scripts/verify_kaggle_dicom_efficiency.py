#!/usr/bin/env python3
"""
scripts/verify_kaggle_dicom_efficiency.py

Kaggle Benchmark Script for Issue 02 (Fast 6-Slot DICOM Preprocessing Pipeline).
Validates on real competition DICOM files:
  1. Header scan vs selective pixel decoding latency per study (assert <= 0.50s / study target).
  2. Rate of InstanceNumber vs physical normal projection (k = p . n) order divergence.
  3. Granular demotion telemetry and failure reason accounting.
  4. Non-finite / NaN absence guard.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

# Set reproducible seeds at script initialization (AGENTS.md standard)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
from src.datasets.efficiency_pipeline import (
    BUDGET_S,
    SLOT_NAMES,
    load_series_metadata,
    parse_dicom_header_fast,
    process_study_to_6slot,
)


def _time_chunk(args: tuple[list[Path], Any, int]) -> float:
    dirs, series_metadata, n_workers = args
    t0 = time.perf_counter()
    for d in dirs:
        process_study_to_6slot(d, series_metadata=series_metadata, num_workers=n_workers)
    return (time.perf_counter() - t0) / max(1, len(dirs))


def benchmark_concurrent(
    studies: list[Path],
    series_metadata: Any,
    n_workers: int,
) -> None:
    """Benchmark real 2-process concurrency under ~4-vCPU Kaggle contention."""
    halves = [studies[0::2], studies[1::2]]
    with ProcessPoolExecutor(max_workers=2) as ex:
        per_study = list(ex.map(_time_chunk, [(h, series_metadata, n_workers) for h in halves]))
    wall = max(per_study)  # seconds per study per process under real contention
    print(f"   Concurrent per-study latency: {wall*1000.0:6.1f} ms -> est. 1300 studies: {(wall * 1300.0 / 2.0) / 60.0:.2f} min (real 2-process 4-vCPU contention)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark RSNA 6-Slot DICOM Preprocessing")
    parser.add_argument("--dicom-root", type=str, default="data/raw", help="Path to raw DICOM studies")
    parser.add_argument("--fingerprints", type=str, default="data/labels/dicom_fingerprints.csv", help="Path to fingerprints CSV")
    parser.add_argument("--series-csv", type=str, default="data/raw/train_series.csv", help="Path to series metadata CSV")
    parser.add_argument("--num-studies", type=int, default=50, help="Number of studies to benchmark")
    parser.add_argument("--num-workers", type=int, default=4, help="Header thread pool workers per study")
    return parser.parse_args()


def benchmark_dicom_pipeline(
    dicom_root: str | Path,
    fingerprints_path: str | Path,
    series_csv_path: str | Path,
    num_studies: int,
    num_workers: int,
) -> None:
    root = Path(dicom_root)
    fp_path = Path(fingerprints_path)
    s_csv_path = Path(series_csv_path)

    # Use shared load_series_metadata loader
    series_metadata = load_series_metadata(s_csv_path)
    if series_metadata is not None:
        print(f"[INFO] Loaded series CSV metadata for {len(series_metadata)} series.")

    studies: list[Path] = []
    if fp_path.exists():
        df_fp = pd.read_csv(fp_path)
        model_col = "ManufacturerModelName" if "ManufacturerModelName" in df_fp.columns else None
        if "StudyInstanceUID" in df_fp.columns and model_col is not None:
            models = df_fp[model_col].dropna().unique()
            per_model = max(1, num_studies // max(1, len(models)))
            sampled_uids = (
                df_fp.groupby(model_col)
                .apply(lambda g: g.head(per_model))
                .reset_index(drop=True)["StudyInstanceUID"]
                .drop_duplicates()
                .tolist()[:num_studies]
            )
            for uid in sampled_uids:
                p = root / uid
                if not p.exists() and (root / "train_series" / uid).exists():
                    p = root / "train_series" / uid
                if p.exists():
                    studies.append(p)

    if not studies:
        candidates = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if not candidates and (root / "train_series").exists():
            candidates = [d for d in (root / "train_series").iterdir() if d.is_dir()]
        studies = candidates[:num_studies]

    if not studies:
        print(f"[WARN] No study directories found under {dicom_root}. Synthetic check recommended.")
        return

    print(f"==================================================================")
    print(f" RSNA Knee MRI: Fast 6-Slot DICOM Preprocessing Benchmark")
    print(f" Studies: {len(studies)} | Workers: {num_workers} | Root: {dicom_root}")
    print(f" Target Latency: <= {BUDGET_S:.2f} s / study")
    print(f" Series CSV Fast-Path: {'Active' if series_metadata is not None else 'Inactive (Physics/Regex)'}")
    print(f"==================================================================")

    k_disagreements = 0
    k_reversals = 0
    total_series_checked = 0
    study_latencies = []
    slot_counts = np.zeros(6, dtype=np.int32)
    aggregated_stats: dict[str, int] = {}
    total_error_types: dict[str, int] = {}

    for i, s_dir in enumerate(studies):
        t0 = time.perf_counter()
        tensor, mask, stats = process_study_to_6slot(
            s_dir,
            series_metadata=series_metadata,
            num_workers=num_workers,
            return_stats=True,
        )
        dt = time.perf_counter() - t0
        study_latencies.append(dt)

        assert tensor.shape == (6, 3, 336, 336), f"Wrong tensor shape: {tensor.shape}"
        assert mask.shape == (6,), f"Wrong mask shape: {mask.shape}"
        assert torch.all(torch.isfinite(tensor)), f"Non-finite values found in {s_dir.name}"

        slot_counts += mask.numpy().astype(np.int32)
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                aggregated_stats[k] = aggregated_stats.get(k, 0) + int(v)
            elif k == "decode_error_types" and isinstance(v, dict):
                for err_k, err_v in v.items():
                    total_error_types[err_k] = total_error_types.get(err_k, 0) + err_v

        # Untimed verification: check k vs InstanceNumber agreement
        dcm_files = [str(f) for f in s_dir.rglob("*") if f.is_file() and not f.name.startswith(".")]
        series_map: dict[str, list] = {}
        for f in dcm_files:
            meta, _ = parse_dicom_header_fast(f, series_metadata=series_metadata)
            if meta:
                series_map.setdefault(meta.series_uid, []).append(meta)

        for _, h_list in series_map.items():
            if len(h_list) >= 3:
                total_series_checked += 1
                by_k = [h.path for h in sorted(h_list, key=lambda x: (x.k_pos, x.instance_number, x.path))]
                by_inst = [h.path for h in sorted(h_list, key=lambda x: x.instance_number)]
                if by_k != by_inst:
                    k_disagreements += 1
                    if by_k == by_inst[::-1]:
                        k_reversals += 1

        print(f"[{i+1:02d}/{len(studies)}] {s_dir.name}: {dt*1000.0:6.1f} ms | Slots: {int(mask.sum().item())}/6")

    mean_dt = float(np.mean(study_latencies))
    p95_dt = float(np.percentile(study_latencies, 95))
    cold_dt = study_latencies[0]
    warm_mean_dt = float(np.mean(study_latencies[1:])) if len(study_latencies) > 1 else cold_dt

    print(f"\n------------------------------------------------------------------")
    print(f" Latency Benchmark Summary (Timed Preprocessing Only):")
    print(f"   Cold Study Latency:  {cold_dt*1000.0:6.1f} ms")
    print(f"   Warm Mean Latency:   {warm_mean_dt*1000.0:6.1f} ms  (Target: <= {BUDGET_S*1000.0:.0f} ms)")
    print(f"   Overall Mean:        {mean_dt*1000.0:6.1f} ms")
    print(f"   P95 Latency:         {p95_dt*1000.0:6.1f} ms")
    print(f"   Estimated 1300 Test Preprocess (Single Process): {(warm_mean_dt * 1300) / 60.0:.2f} minutes")
    print(f"   Estimated 1300 Test Preprocess (Dual GPU Pool, Best Case): {(warm_mean_dt * 1300 / 2.0) / 60.0:.2f} minutes")
    if len(studies) >= 2:
        print(f"\n Dual-Process Contention Benchmark (2 Concurrent Workers):")
        benchmark_concurrent(studies, series_metadata, num_workers)
    print(f"\n Spatial Sorting Analysis:")
    print(f"   Total Series:        {total_series_checked}")
    print(f"   k != InstanceNumber: {k_disagreements} ({(k_disagreements/max(1, total_series_checked))*100.0:.1f}%)")
    print(f"   Exact Inversions:    {k_reversals} ({(k_reversals/max(1, total_series_checked))*100.0:.1f}%)")
    print(f"\n Granular Header & Demotion Telemetry:")
    for k, v in sorted(aggregated_stats.items()):
        print(f"   {k:<22}: {v}")
    if total_error_types:
        print(f"\n Decode Error Breakdown:")
        for err_k, err_cnt in total_error_types.items():
            print(f"   {err_k:<22}: {err_cnt}")
    print(f"\n Slot Presence Distribution:")
    for slot_idx, name in enumerate(SLOT_NAMES):
        pct = (slot_counts[slot_idx] / len(studies)) * 100.0
        print(f"   Slot {slot_idx} ({name:<16}): {slot_counts[slot_idx]}/{len(studies)} ({pct:5.1f}%)")
    print(f"==================================================================")

    assert aggregated_stats.get("decode_errors", 0) == 0, (
        f"Detected {aggregated_stats.get('decode_errors')} decoding errors during benchmark; check DICOM codecs!"
    )
    assert warm_mean_dt <= BUDGET_S, f"Preprocessing exceeds efficiency budget: {warm_mean_dt:.3f}s > {BUDGET_S:.3f}s"


if __name__ == "__main__":
    args = parse_args()
    benchmark_dicom_pipeline(
        args.dicom_root,
        args.fingerprints,
        args.series_csv,
        args.num_studies,
        args.num_workers,
    )
