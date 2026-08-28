"""
orchestrator_agent.py -- AGY Orchestrator Agent for RSNA Phase 2.

v4 Design:
  - Reads gemini_model_primary and gemini_model_fallback from config.yaml.
  - Passes both model identifiers to run_extractor_agent() as explicit arguments.
  - Dynamic banner print ({primary_model} primary).
  - Strict gold study protection during imputation.
  - No hardcoded model constants anywhere in this file.
  - Single source of truth for model selection: config.yaml nlp: section.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from pathlib import Path

import pandas as pd
import yaml

from .extractor_agent import run_extractor_agent
from .imputer import apply_silence_imputation
from .tools import (
    export_language_tags,
    get_total_batches,
    load_unlabelled_batch,
    merge_and_save_pseudo_labels,
)
from .validator_agent import compute_concordance

logger = logging.getLogger(__name__)


_ORCHESTRATOR_SYSTEM_PROMPT = """You are the Phase 2 Orchestration Agent for the RSNA 2026
Knee Abnormality Detection challenge. Coordinate parallel LLM-based weak label extraction
from 4,349 multilingual radiology reports.

GOVERNANCE RULES:
1. Never overwrite the 58 gold study rows -- image-derived labels are authoritative.
2. Use -1 as the safe fallback (not 0) -- 0 means explicit negative, -1 means unaddressed.
3. Run the Validator subagent EXACTLY ONCE after all extractors finish.
4. Report macro AUC and the 82.5% structural ceiling to the user.
5. Log escalation counts: Flash-primary / Flash-fallback / hard-fallback breakdown.
"""


async def run_orchestrator(
    config_path: str = "config.yaml",
    api_key: str | None = None,
    dry_run_n: int | None = None,
    skip_validation: bool = False,
    n_concurrent_workers: int = 8,
) -> None:
    """Launch the full Phase 2 extraction pipeline.

    Args:
        config_path: Path to project config.yaml.
        api_key: Gemini API key. If None, reads GEMINI_API_KEY env var.
        dry_run_n: If set, extract only the first N studies (testing).
        skip_validation: Skip Validator subagent (for incremental re-runs).
        n_concurrent_workers: Concurrent Extractor agents (default 8).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["project"]["root"])
    train_csv = root / cfg["data"]["train_csv"]
    cache_dir = root / cfg["nlp"]["cache_dir"]
    output_csv = root / cfg["nlp"]["pseudo_labels_csv"]
    concordance_json = root / cfg["nlp"]["concordance_json"]
    lang_csv = root / cfg["nlp"]["lang_csv"]
    batch_size = cfg["nlp"].get("batch_size", 50)

    # Model identifiers read from config.yaml -- never hardcoded.
    primary_model  = cfg["nlp"].get("gemini_model_primary",  "gemini-3.7-flash")
    fallback_model = cfg["nlp"].get("gemini_model_fallback", "gemini-3.6-flash")

    cache_dir.mkdir(parents=True, exist_ok=True)
    api_key = api_key or os.environ.get("GEMINI_API_KEY")

    # -----------------------------------------------------------------------
    # Step 0: Export language tags
    # -----------------------------------------------------------------------
    logger.info("Step 0: Checking language_tags.csv...")
    lang_status = export_language_tags(str(train_csv), str(lang_csv))
    logger.info(f"Language tags: {lang_status}")

    # -----------------------------------------------------------------------
    # Step 1: Calculate total work
    # -----------------------------------------------------------------------
    total_batches = int(get_total_batches(str(train_csv), str(cache_dir), batch_size))

    if dry_run_n:
        total_batches = min(total_batches, math.ceil(dry_run_n / batch_size))
        logger.info(f"DRY RUN: limiting to {dry_run_n} studies ({total_batches} batches)")

    logger.info(
        f"Total batches: {total_batches} × {batch_size} studies | "
        f"Workers: {n_concurrent_workers} | "
        f"Primary model: {primary_model} | Fallback model: {fallback_model}"
    )

    # -----------------------------------------------------------------------
    # Step 2: Load batch data
    # -----------------------------------------------------------------------
    batches: list[list[dict]] = []
    if total_batches > 0:
        for i in range(total_batches):
            batch_json = load_unlabelled_batch(str(train_csv), str(cache_dir), i, batch_size)
            batch_data = json.loads(batch_json)
            if batch_data:
                batches.append(batch_data)
        logger.info(f"Loaded {len(batches)} batches ({sum(len(b) for b in batches)} studies)")

    # -----------------------------------------------------------------------
    # Step 3: Concurrent extractor subagents
    # -----------------------------------------------------------------------
    all_results = []
    if batches:
        semaphore = asyncio.Semaphore(n_concurrent_workers)

        async def run_with_semaphore(batch_data: list[dict], batch_id: int):
            async with semaphore:
                return await run_extractor_agent(
                    batch=batch_data,
                    batch_id=batch_id,
                    cache_dir=cache_dir,
                    api_key=api_key,
                    primary_model=primary_model,
                    fallback_model=fallback_model,
                )

        tasks = [
            asyncio.create_task(run_with_semaphore(bd, i))
            for i, bd in enumerate(batches)
        ]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            all_results.append(result)
            if len(all_results) % 10 == 0:
                logger.info(f"Incremental save ({len(all_results)}/{len(tasks)} batches)...")
                merge_and_save_pseudo_labels(str(train_csv), str(cache_dir), str(output_csv))

    # Aggregate escalation stats
    total_flash = sum(
        sum(1 for r in br.results if r.extraction_source == "extractor_flash")
        for br in all_results
    )
    total_escalated = sum(r.studies_escalated for r in all_results)
    total_failed = sum(r.studies_failed for r in all_results)
    total_cached = sum(
        sum(1 for r in br.results if r.extraction_source == "cache")
        for br in all_results
    )

    logger.info(
        f"\nExtraction summary:\n"
        f"  Flash primary success:  {total_flash}\n"
        f"  Escalated to fallback:  {total_escalated}\n"
        f"  Hard fallback (all -1): {total_failed}\n"
        f"  Cache hits:             {total_cached}"
    )

    # -----------------------------------------------------------------------
    # Step 4: Imputation + final save
    # -----------------------------------------------------------------------
    logger.info("Applying silence imputation rules...")
    summary = merge_and_save_pseudo_labels(str(train_csv), str(cache_dir), str(output_csv))
    logger.info(f"Merge: {summary}")

    # Extract gold UIDs for strict protection during imputation
    gold_df = pd.read_csv(train_csv, on_bad_lines="skip")
    label_cols = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA",
        "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
    ]
    existing_cols = [c for c in label_cols if c in gold_df.columns]
    gold_uids = set(gold_df[gold_df[existing_cols].notna().any(axis=1)]["StudyInstanceUID"].astype(str))

    pseudo_df = pd.read_csv(output_csv)
    pseudo_df = apply_silence_imputation(pseudo_df, gold_uids=gold_uids)
    pseudo_df.to_csv(output_csv, index=False)
    logger.info(f"Imputation applied. Final rows: {len(pseudo_df)}")

    # -----------------------------------------------------------------------
    # Step 5: Validator (once -- governance rule)
    # -----------------------------------------------------------------------
    if not skip_validation:
        logger.info("Launching Validator subagent (runs ONCE)...")
        report = compute_concordance(
            pseudo_labels_csv=str(output_csv),
            train_csv_path=str(train_csv),
            lang_csv_path=str(lang_csv) if lang_csv.exists() else None,
            output_json_path=str(concordance_json),
        )
        print("\n" + "=" * 62)
        print(f"  PHASE 2 CONCORDANCE REPORT  ({primary_model} primary)")
        print("=" * 62)
        print(f"  Macro AUC (vs 58 gold):   {report.macro_auc:.4f}")
        print(f"  Overall PPV:              {report.overall_ppv:.4f}")
        print(f"  Overall Recall:           {report.overall_recall:.4f}")
        print(f"  Gold studies evaluated:   {report.gold_studies_evaluated}")
        print(f"\n  Per-Label AUC:")
        for lbl, auc in report.per_label_auc.items():
            print(f"    {lbl:<25} {auc:.4f}")
        if report.per_language_auc:
            print(f"\n  Per-Language Macro AUC:")
            for lang, auc in sorted(report.per_language_auc.items(), key=lambda x: -x[1]):
                print(f"    {lang:<8} {auc:.4f}")
        print(f"\n  Top Silence Rates (post-extraction, pre-imputation):")
        top5 = sorted(report.silence_rates.items(), key=lambda x: -x[1])[:5]
        for lbl, rate in top5:
            print(f"    {lbl:<25} {rate*100:.1f}%")
        print(f"\n    {report.concordance_ceiling_note}")
        print(f"\n  Extraction breakdown:  "
              f"{total_flash} flash | {total_escalated} escalated | {total_failed} hard-fail")
        print("=" * 62)
    else:
        logger.info("Skipping validation (--skip-validation flag).")

    logger.info("Phase 2 complete.")
