"""
extractor_agent.py -- AGY Extractor Subagent for RSNA Phase 2.

v4 Design:
  - Model identifiers are dynamically passed to run_extractor_agent() from config.yaml.
  - Safe defaults (_DEFAULT_PRIMARY_MODEL / _DEFAULT_FALLBACK_MODEL) are provided.
  - Real two-tier escalation: gemini-3.7-flash → gemini-3.6-flash → hard-fallback (-1).
  - extraction_source field records: extractor_flash / extractor_flash_fallback / hard_fallback / cache.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from google.antigravity import Agent, LocalAgentConfig, types
from google.antigravity.hooks import hooks

from .schemas import KneeLabelExtraction, StudyExtractionResult, StudyLabelBatch

logger = logging.getLogger(__name__)


_EXTRACTION_SYSTEM_PROMPT = """You are an expert subspecialty musculoskeletal (MSK) radiologist.
Your ONLY job is to read a knee MRI radiology report (written in ANY language) and extract
the exact clinical status for each of the 12 target findings listed below.

CRITICAL ANNOTATION RULES (Official RSNA Radiologist Ground-Truth Guidelines):
1. Golden Rule: Borderline, mild, or ambiguous findings MUST be labeled as 0 (favor high specificity).
2. acl_tear: 1 ONLY if high-grade partial (>50% fibers disrupted) or full-thickness tear/rupture.
   Mild degeneration or intact reconstruction = 0.
3. mcl_tear: 1 ONLY if high-grade partial or complete tear. Low-grade sprain (Grade 1/2) = 0.
4. medial_meniscus: 1 ONLY if tear contacts articular surface on ≥2 consecutive images, or shows
   displacement/flap/bucket-handle. Intrasubstance degeneration = 0.
5. lateral_meniscus: Same criteria as medial meniscus. Intrasubstance degeneration = 0.
6. medial_oa: 1 ONLY if moderate-to-severe (>1cm area with >50% cartilage thickness loss).
   Mild thinning = 0.
7. lateral_oa: Same criteria as medial OA.
8. patellofemoral_oa: Same criteria as medial OA.
9. joint_effusion: 1 ONLY if moderate or large fluid distension. Trace or physiological fluid = 0.
10. synovitis: 1 ONLY if marked synovial thickening or inflammatory hypertrophy is described.
11. bakers_cyst: 1 ONLY if moderate or large popliteal cyst. Tiny incidental bursa fluid = 0.
12. bone_contusion: 1 ONLY for traumatic marrow edema / impact contusion.
    Chronic degenerative OA edema = 0.
13. fracture: 1 ONLY for acute cortical fracture line. Healed remote fracture = 0.

OUTPUT VALUES (use EXACTLY these integers -- no strings, no decimals):
   1  = The report EXPLICITLY states this finding is present / positive.
   0  = The report EXPLICITLY states this finding is absent / normal / negative.
  -1  = The report does NOT address or mention this finding at all.
        DO NOT use 0 for silence -- use -1. Reserve 0 ONLY for explicit negations.

Output a JSON object with exactly these 12 keys. No explanatory text.
"""

_DEFAULT_PRIMARY_MODEL  = "gemini-3.7-flash"
_DEFAULT_FALLBACK_MODEL = "gemini-3.6-flash"


def _build_extraction_prompt(study_uid: str, report_text: str) -> str:
    return (
        f"StudyInstanceUID: {study_uid}\n\n"
        f"Radiology Report:\n{report_text}\n\n"
        "Extract the 12-label JSON for this report now."
    )


def _make_extractor_config(model: str, api_key: str | None) -> LocalAgentConfig:
    """Build an AGY LocalAgentConfig for an extractor agent."""
    return LocalAgentConfig(
        model=model,
        system_instructions=_EXTRACTION_SYSTEM_PROMPT,
        response_schema=KneeLabelExtraction,  # SDK enforces 3-state schema
        capabilities=types.CapabilitiesConfig(
            enable_subagents=False,  # Extractors are leaf workers
        ),
        **({"api_key": api_key} if api_key else {}),
    )


async def _extract_single_study(
    agent: Agent,
    study_uid: str,
    report_text: str,
) -> KneeLabelExtraction | None:
    """Attempt to extract labels for one study using the provided agent."""
    try:
        prompt = _build_extraction_prompt(study_uid, report_text)
        response = await agent.chat(prompt)
        structured = await response.structured_output()
        if structured is None:
            logger.warning(f"[{study_uid}] structured_output() returned None")
            return None
        return KneeLabelExtraction(**structured)
    except Exception as e:
        logger.warning(f"[{study_uid}] Extraction attempt failed: {e}")
        return None


async def run_extractor_agent(
    batch: list[dict],
    batch_id: int,
    cache_dir: Path,
    api_key: str | None = None,
    primary_model: str = _DEFAULT_PRIMARY_MODEL,
    fallback_model: str = _DEFAULT_FALLBACK_MODEL,
) -> StudyLabelBatch:
    """Run extraction on one batch of studies with two-tier model escalation.

    Tier 1: primary_model  (default: gemini-3.7-flash -- read from config.yaml)
    Tier 2: fallback_model (default: gemini-3.6-flash -- read from config.yaml)
    Tier 3: Hard fallback all-(-1) -- only if both tiers fail
    """
    results: list[StudyExtractionResult] = []
    studies_failed = 0
    studies_escalated = 0

    # Partition batch into cache hits and studies needing extraction
    cache_hits = []
    to_extract = []
    for study in batch:
        uid = study["study_uid"]
        cache_file = cache_dir / f"{uid}.json"
        if cache_file.exists():
            cache_hits.append(study)
        else:
            to_extract.append(study)

    # Serve cache hits immediately
    for study in cache_hits:
        uid = study["study_uid"]
        cached_data = json.loads((cache_dir / f"{uid}.json").read_text(encoding="utf-8"))
        results.append(StudyExtractionResult(
            study_uid=uid,
            labels=KneeLabelExtraction(**cached_data),
            extraction_source="cache",
        ))

    if not to_extract:
        return StudyLabelBatch(
            results=results,
            batch_id=batch_id,
            studies_processed=len(batch),
            studies_failed=0,
            studies_escalated=0,
        )

    # Tier 1: primary_model (from config.yaml) -- process entire batch in one session
    primary_config = _make_extractor_config(primary_model, api_key)
    tier1_failures: list[dict] = []

    try:
        async with Agent(primary_config) as primary_agent:
            for study in to_extract:
                uid = study["study_uid"]
                labels = await _extract_single_study(primary_agent, uid, study["report_text"])

                if labels is not None:
                    # Success -- write to cache and record
                    cache_file = cache_dir / f"{uid}.json"
                    tmp = cache_file.with_suffix(".tmp")
                    tmp.write_text(labels.model_dump_json(), encoding="utf-8")
                    tmp.rename(cache_file)
                    results.append(StudyExtractionResult(
                        study_uid=uid,
                        labels=labels,
                        extraction_source="extractor_flash",
                    ))
                else:
                    # Tier 1 failure -- queue for Tier 2
                    tier1_failures.append(study)
    except Exception as e:
        logger.warning(f"[Batch {batch_id}] Primary model session error: {e}")
        extracted_uids = {r.study_uid for r in results}
        tier1_failures = [s for s in to_extract if s["study_uid"] not in extracted_uids]

    # Tier 2: fallback_model (from config.yaml) -- retry only Tier 1 failures
    tier2_failures: list[dict] = []
    if tier1_failures:
        logger.info(
            f"[Batch {batch_id}] {len(tier1_failures)} studies escalating to "
            f"{fallback_model} fallback tier"
        )
        fallback_config = _make_extractor_config(fallback_model, api_key)

        try:
            async with Agent(fallback_config) as fallback_agent:
                for study in tier1_failures:
                    uid = study["study_uid"]
                    labels = await _extract_single_study(fallback_agent, uid, study["report_text"])

                    if labels is not None:
                        # Tier 2 success
                        cache_file = cache_dir / f"{uid}.json"
                        tmp = cache_file.with_suffix(".tmp")
                        tmp.write_text(labels.model_dump_json(), encoding="utf-8")
                        tmp.rename(cache_file)
                        results.append(StudyExtractionResult(
                            study_uid=uid,
                            labels=labels,
                            extraction_source="extractor_flash_fallback",
                        ))
                        studies_escalated += 1
                    else:
                        tier2_failures.append(study)
        except Exception as e:
            logger.warning(f"[Batch {batch_id}] Fallback model session error: {e}")
            extracted_uids = {r.study_uid for r in results}
            tier2_failures = [s for s in tier1_failures if s["study_uid"] not in extracted_uids]

    # Tier 3: hard fallback -- all -1
    if tier2_failures:
        for study in tier2_failures:
            uid = study["study_uid"]
            logger.warning(
                f"[Batch {batch_id}] Hard fallback (all -1) for {uid}: "
                f"both {primary_model} and {fallback_model} failed."
            )
            fallback_labels = KneeLabelExtraction(
                **{k: -1 for k in KneeLabelExtraction.model_fields}
            )
            cache_file = cache_dir / f"{uid}.json"
            tmp = cache_file.with_suffix(".tmp")
            tmp.write_text(fallback_labels.model_dump_json(), encoding="utf-8")
            tmp.rename(cache_file)
            results.append(StudyExtractionResult(
                study_uid=uid,
                labels=fallback_labels,
                extraction_source="hard_fallback",
            ))
            studies_failed += 1

    logger.info(
        f"[Batch {batch_id}] Done: "
        f"{len(results) - studies_failed - studies_escalated} flash | "
        f"{studies_escalated} escalated-to-fallback | "
        f"{studies_failed} hard-fallback"
    )

    return StudyLabelBatch(
        results=results,
        batch_id=batch_id,
        studies_processed=len(batch),
        studies_failed=studies_failed,
        studies_escalated=studies_escalated,
    )
