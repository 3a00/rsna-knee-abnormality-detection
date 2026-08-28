"""
tools.py -- Tool functions available to the Orchestrator agent.

Functions for cache management, batch loading, CSV merging, and
language tag exports. All paths are passed as arguments -- never hardcoded.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Step 0 tool: Language tags export
# ---------------------------------------------------------------------------

def export_language_tags(train_csv_path: str, output_csv_path: str) -> str:
    """Persist per-study language detection tags to disk for Validator subagent.

    If the output file already exists, this is a no-op (returns 'ALREADY_EXISTS').
    Otherwise, runs langdetect on all 4,407 reports and saves:
      columns: StudyInstanceUID, detected_lang (ISO 639-1 code or 'unknown')

    Args:
        train_csv_path: Absolute path to train.csv.
        output_csv_path: Where to write language_tags.csv.

    Returns:
        'ALREADY_EXISTS' if file was already present, or a summary string on completion.
    """
    output_path = Path(output_csv_path)
    if output_path.exists():
        return f"ALREADY_EXISTS: {output_path} ({output_path.stat().st_size} bytes)"

    try:
        from langdetect import detect, LangDetectException
    except ImportError:
        return "ERROR: langdetect not installed. Run: pip install langdetect"

    def _detect(text: str) -> str:
        try:
            if pd.isna(text) or len(str(text).strip()) < 20:
                return "unknown"
            return detect(str(text))
        except Exception:
            return "unknown"

    df = pd.read_csv(train_csv_path, on_bad_lines="skip")
    df["detected_lang"] = df["Report"].apply(_detect)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lang_df = df[["StudyInstanceUID", "detected_lang"]]
    lang_df.to_csv(output_path, index=False)

    dist = df["detected_lang"].value_counts().head(10).to_dict()
    return f"SAVED: {len(lang_df)} rows → {output_path} | top languages: {dist}"


# ---------------------------------------------------------------------------
# Cache tools
# ---------------------------------------------------------------------------

def check_cache(cache_dir: str, study_uid: str) -> str:
    """Check whether a study's extraction result is already cached on disk.

    Returns the cached JSON string if found, or 'CACHE_MISS' if not.

    Args:
        cache_dir: Absolute path to the cache directory (data/labels/cache/).
        study_uid: StudyInstanceUID string to look up.
    """
    cache_path = Path(cache_dir) / f"{study_uid}.json"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    return "CACHE_MISS"


def save_to_cache(cache_dir: str, study_uid: str, extraction_json: str) -> str:
    """Persist one study's extraction result to the disk cache atomically.

    Writes temp file then renames to prevent partial-write corruption on crash.

    Args:
        cache_dir: Absolute path to the cache directory.
        study_uid: StudyInstanceUID string (used as filename).
        extraction_json: JSON string matching KneeLabelExtraction schema.

    Returns:
        'OK' on success, or an error description on failure.
    """
    cache_path = Path(cache_dir) / f"{study_uid}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    try:
        tmp_path.write_text(extraction_json, encoding="utf-8")
        tmp_path.rename(cache_path)
        return "OK"
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Batch loading tools
# ---------------------------------------------------------------------------

def load_unlabelled_batch(
    train_csv_path: str,
    cache_dir: str,
    batch_index: int,
    batch_size: int,
) -> str:
    """Load one batch of unlabelled study reports, skipping cache hits and gold rows.

    Args:
        train_csv_path: Absolute path to train.csv.
        cache_dir: Absolute path to the cache directory.
        batch_index: Zero-based batch index.
        batch_size: Studies per batch (typically 50).

    Returns:
        JSON list of {"study_uid": str, "report_text": str} dicts.
        Returns '[]' if batch_index is out of range (signals orchestrator to stop).
    """
    LABEL_COLS = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA",
        "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
    ]
    df = pd.read_csv(train_csv_path, on_bad_lines="skip")
    label_cols = [c for c in LABEL_COLS if c in df.columns]
    gold_mask = df[label_cols].notna().any(axis=1)
    unlabelled = df[~gold_mask].copy()

    cache_path = Path(cache_dir)
    uncached = unlabelled[
        ~unlabelled["StudyInstanceUID"].apply(
            lambda uid: (cache_path / f"{uid}.json").exists()
        )
    ].reset_index(drop=True)

    start = batch_index * batch_size
    batch = uncached.iloc[start: start + batch_size]
    if batch.empty:
        return "[]"

    records = [
        {"study_uid": str(row["StudyInstanceUID"]), "report_text": str(row["Report"])}
        for _, row in batch.iterrows()
    ]
    return json.dumps(records, ensure_ascii=False)


def get_total_batches(train_csv_path: str, cache_dir: str, batch_size: int) -> str:
    """Return the total number of uncached unlabelled batches remaining.

    Args:
        train_csv_path: Absolute path to train.csv.
        cache_dir: Absolute path to the cache directory.
        batch_size: Studies per batch.

    Returns:
        String integer batch count.
    """
    LABEL_COLS = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA",
        "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
    ]
    df = pd.read_csv(train_csv_path, on_bad_lines="skip")
    label_cols = [c for c in LABEL_COLS if c in df.columns]
    gold_mask = df[label_cols].notna().any(axis=1)
    unlabelled = df[~gold_mask].copy()
    cache_path = Path(cache_dir)
    uncached_count = unlabelled[
        ~unlabelled["StudyInstanceUID"].apply(
            lambda uid: (cache_path / f"{uid}.json").exists()
        )
    ].shape[0]
    return str(math.ceil(uncached_count / batch_size))


# ---------------------------------------------------------------------------
# Merge & output tools
# ---------------------------------------------------------------------------

def merge_and_save_pseudo_labels(
    train_csv_path: str,
    cache_dir: str,
    output_csv_path: str,
) -> str:
    """Merge all cached extraction results with gold labels → pseudo_labels.csv.

    Gold rows (n=58): labels taken verbatim from train.csv (never overwritten).
    Unlabelled rows (n=4,349): labels from cache JSONs; NaN if no cache entry.

    Returns:
        Summary string with row counts.

    Raises:
        AssertionError: If any gold row label values were modified (critical guard).
    """
    LABEL_COLUMNS = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA",
        "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
    ]
    PROMPT_KEY_TO_CSV_COL = {
        "acl_tear": "ACL", "mcl_tear": "MCL",
        "medial_meniscus": "Medial Meniscus", "lateral_meniscus": "Lateral Meniscus",
        "medial_oa": "Medial OA", "lateral_oa": "Lateral OA",
        "patellofemoral_oa": "PF OA", "joint_effusion": "Effusion",
        "synovitis": "Synovitis", "bakers_cyst": "Baker's",
        "bone_contusion": "Contusion", "fracture": "Fracture",
    }

    df = pd.read_csv(train_csv_path, on_bad_lines="skip")
    original_df = df.copy()
    label_cols = [c for c in LABEL_COLUMNS if c in df.columns]
    gold_mask = df[label_cols].notna().any(axis=1)
    cache_path = Path(cache_dir)
    extracted_count, missing_count = 0, 0

    for idx, row in df[~gold_mask].iterrows():
        uid = str(row["StudyInstanceUID"])
        cache_file = cache_path / f"{uid}.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            for prompt_key, csv_col in PROMPT_KEY_TO_CSV_COL.items():
                if csv_col in df.columns:
                    df.at[idx, csv_col] = data.get(prompt_key, float("nan"))
            extracted_count += 1
        else:
            missing_count += 1

    # Critical safety assertion: gold rows must be untouched
    gold_after = df[gold_mask][label_cols]
    gold_before = original_df[gold_mask][label_cols]
    assert gold_after.equals(gold_before), (
        "CRITICAL: Gold labels were modified! This is a data integrity violation. "
        "Abort and investigate before saving."
    )

    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    return (
        f"pseudo_labels.csv saved: {len(df)} total | "
        f"{gold_mask.sum()} gold (verbatim) | "
        f"{extracted_count} extracted | "
        f"{missing_count} missing (NaN)"
    )


def load_gold_studies(train_csv_path: str) -> str:
    """Load the 58 gold studies with their authoritative labels as JSON."""
    LABEL_COLS = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA",
        "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
    ]
    df = pd.read_csv(train_csv_path, on_bad_lines="skip")
    label_cols = [c for c in LABEL_COLS if c in df.columns]
    gold = df[df[label_cols].notna().any(axis=1)].copy()
    return gold.to_json(orient="records", force_ascii=False)
