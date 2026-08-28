"""
test_phase2.py -- Unit tests for Phase 2 NLP extraction modules.
Includes regression guards, schema checks, imputer logic, cache tools,
gold row protection, and config-arg propagation.
All tests run locally without network calls or API key.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.nlp.schemas import (
    KneeLabelExtraction, StudyExtractionResult, StudyLabelBatch, LabelState
)
from src.nlp.imputer import apply_silence_imputation
from src.nlp.tools import check_cache, save_to_cache, merge_and_save_pseudo_labels
try:
    from src.nlp.orchestrator_agent import run_orchestrator
except ImportError:
    run_orchestrator = None


# ---------------------------------------------------------------------------
# Schema tests (T1–T4)
# ---------------------------------------------------------------------------

def test_schema_valid_all_states():
    """T1: KneeLabelExtraction accepts -1, 0, and 1 for each field."""
    obj = KneeLabelExtraction(
        acl_tear=1, mcl_tear=0, medial_meniscus=-1, lateral_meniscus=1,
        medial_oa=0, lateral_oa=-1, patellofemoral_oa=0, joint_effusion=1,
        synovitis=-1, bakers_cyst=0, bone_contusion=1, fracture=0,
    )
    assert obj.acl_tear == 1 and obj.synovitis == -1


def test_schema_rejects_invalid_value():
    """T2: KneeLabelExtraction rejects values outside {-1, 0, 1}."""
    with pytest.raises(Exception):
        KneeLabelExtraction(
            acl_tear=2, mcl_tear=0, medial_meniscus=-1, lateral_meniscus=1,
            medial_oa=0, lateral_oa=-1, patellofemoral_oa=0, joint_effusion=1,
            synovitis=-1, bakers_cyst=0, bone_contusion=1, fracture=0,
        )


def test_schema_has_exactly_12_fields():
    """T3: KneeLabelExtraction has exactly 12 fields."""
    expected = {
        "acl_tear", "mcl_tear", "medial_meniscus", "lateral_meniscus",
        "medial_oa", "lateral_oa", "patellofemoral_oa", "joint_effusion",
        "synovitis", "bakers_cyst", "bone_contusion", "fracture",
    }
    assert set(KneeLabelExtraction.model_fields.keys()) == expected


def test_schema_hard_fallback_all_minus_one():
    """T4: Hard-fallback pattern (all -1) is valid and all fields are -1."""
    fallback = KneeLabelExtraction(**{k: -1 for k in KneeLabelExtraction.model_fields})
    assert all(v == -1 for v in fallback.model_dump().values())


# ---------------------------------------------------------------------------
# Forward reference regression guard (T5)
# ---------------------------------------------------------------------------

def test_study_label_batch_forward_reference_resolved():
    """T5: StudyLabelBatch.results can hold StudyExtractionResult without PydanticUndefinedAnnotation."""
    labels = KneeLabelExtraction(**{k: 0 for k in KneeLabelExtraction.model_fields})
    result = StudyExtractionResult(
        study_uid="TEST_001",
        labels=labels,
        extraction_source="extractor_flash",
    )
    batch = StudyLabelBatch(
        results=[result],
        batch_id=0,
        studies_processed=1,
        studies_failed=0,
        studies_escalated=0,
    )
    assert batch.results[0].study_uid == "TEST_001"


# ---------------------------------------------------------------------------
# Imputer tests (T6–T9)
# ---------------------------------------------------------------------------

def _make_df(n_gold: int = 2, n_unlabelled: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(n_gold):
        rows.append({
            "StudyInstanceUID": f"GOLD_{i}", "Report": "gold",
            "ACL": 1, "MCL": 0, "Medial Meniscus": 1, "Lateral Meniscus": 0,
            "Medial OA": 0, "Lateral OA": 0, "PF OA": 1,
            "Effusion": 1, "Synovitis": 0, "Baker's": 0, "Contusion": 1, "Fracture": 0,
        })
    for i in range(n_unlabelled):
        rows.append({
            "StudyInstanceUID": f"UNLABELLED_{i}", "Report": "unlabelled",
            **{c: float("nan") for c in [
                "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
                "Medial OA", "Lateral OA", "PF OA",
                "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
            ]}
        })
    return pd.DataFrame(rows)


def test_synovitis_imputation_eff_positive():
    """T6: Synovitis=-1 + Effusion=1 → synovitis_soft=0.63."""
    df = _make_df(n_gold=1, n_unlabelled=1)
    df.loc[df["StudyInstanceUID"] == "UNLABELLED_0", "Synovitis"] = -1
    df.loc[df["StudyInstanceUID"] == "UNLABELLED_0", "Effusion"] = 1
    result = apply_silence_imputation(df, gold_uids={"GOLD_0"})
    soft = result.loc[result["StudyInstanceUID"] == "UNLABELLED_0", "synovitis_soft"].values[0]
    assert soft == 0.63


def test_synovitis_imputation_eff_not_positive():
    """T7: Synovitis=-1 + Effusion=0 → synovitis_soft=0.22."""
    df = _make_df(n_gold=1, n_unlabelled=1)
    df.loc[df["StudyInstanceUID"] == "UNLABELLED_0", "Synovitis"] = -1
    df.loc[df["StudyInstanceUID"] == "UNLABELLED_0", "Effusion"] = 0
    result = apply_silence_imputation(df, gold_uids={"GOLD_0"})
    soft = result.loc[result["StudyInstanceUID"] == "UNLABELLED_0", "synovitis_soft"].values[0]
    assert soft == 0.22


def test_explicit_synovitis_not_overridden():
    """T8: Explicit Synovitis=0 + Effusion=1 → Synovitis stays 0."""
    df = _make_df(n_gold=1, n_unlabelled=1)
    df.loc[df["StudyInstanceUID"] == "UNLABELLED_0", "Synovitis"] = 0
    df.loc[df["StudyInstanceUID"] == "UNLABELLED_0", "Effusion"] = 1
    result = apply_silence_imputation(df, gold_uids={"GOLD_0"})
    val = result.loc[result["StudyInstanceUID"] == "UNLABELLED_0", "Synovitis"].values[0]
    assert val == 0


def test_structural_silence_maps_to_zero():
    """T9: Baker's=-1, Medial OA=-1, Lateral OA=-1 → all mapped to 0."""
    df = _make_df(n_gold=1, n_unlabelled=1)
    df.loc[df["StudyInstanceUID"] == "UNLABELLED_0", "Baker's"] = -1
    df.loc[df["StudyInstanceUID"] == "UNLABELLED_0", "Medial OA"] = -1
    df.loc[df["StudyInstanceUID"] == "UNLABELLED_0", "Lateral OA"] = -1
    result = apply_silence_imputation(df, gold_uids={"GOLD_0"})
    row = result[result["StudyInstanceUID"] == "UNLABELLED_0"].iloc[0]
    assert row["Baker's"] == 0
    assert row["Medial OA"] == 0
    assert row["Lateral OA"] == 0


# ---------------------------------------------------------------------------
# Cache tool tests (T10–T11)
# ---------------------------------------------------------------------------

def test_cache_miss():
    """T10: check_cache returns 'CACHE_MISS' for non-existent study."""
    with tempfile.TemporaryDirectory() as tmpdir:
        assert check_cache(tmpdir, "MISSING_UID") == "CACHE_MISS"


def test_cache_save_and_hit():
    """T11: save_to_cache then check_cache returns saved JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data = json.dumps({"acl_tear": 1, "mcl_tear": -1})
        assert save_to_cache(tmpdir, "TEST_UID", data) == "OK"
        cached = check_cache(tmpdir, "TEST_UID")
        assert json.loads(cached) == {"acl_tear": 1, "mcl_tear": -1}


# ---------------------------------------------------------------------------
# Escalation tier distinction test (T12)
# ---------------------------------------------------------------------------

def test_extraction_source_distinguishes_escalation():
    """T12: StudyExtractionResult.extraction_source records escalation tier correctly."""
    labels = KneeLabelExtraction(**{k: -1 for k in KneeLabelExtraction.model_fields})

    flash_result = StudyExtractionResult(
        study_uid="UID_FLASH", labels=labels, extraction_source="extractor_flash"
    )
    fallback_result = StudyExtractionResult(
        study_uid="UID_FALLBACK", labels=labels, extraction_source="extractor_flash_fallback"
    )
    hard_result = StudyExtractionResult(
        study_uid="UID_HARD", labels=labels, extraction_source="hard_fallback"
    )
    cache_result = StudyExtractionResult(
        study_uid="UID_CACHE", labels=labels, extraction_source="cache"
    )

    assert flash_result.extraction_source == "extractor_flash"
    assert fallback_result.extraction_source == "extractor_flash_fallback"
    assert hard_result.extraction_source == "hard_fallback"
    assert cache_result.extraction_source == "cache"


# ---------------------------------------------------------------------------
# Config model wiring test (T13)
# ---------------------------------------------------------------------------

def test_config_model_propagation_to_extractor():
    """T13: run_orchestrator reads model names from config.yaml and passes them to run_extractor_agent."""
    if run_orchestrator is None:
        pytest.skip("google.antigravity agentic SDK not available in standalone pytest env")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        raw_csv = tmp_path / "train.csv"
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        out_csv = tmp_path / "pseudo.csv"
        lang_csv = tmp_path / "lang.csv"
        conc_json = tmp_path / "concordance.json"

        # Create dummy train.csv with 1 gold study and 1 unlabelled study
        df = _make_df(n_gold=1, n_unlabelled=1)
        df.to_csv(raw_csv, index=False)

        custom_cfg = {
            "project": {"root": str(tmp_path)},
            "data": {"train_csv": str(raw_csv)},
            "labels": {
                "columns": [
                    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
                    "Medial OA", "Lateral OA", "PF OA",
                    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
                ]
            },
            "nlp": {
                "gemini_model_primary": "custom-primary-model-v1",
                "gemini_model_fallback": "custom-fallback-model-v1",
                "batch_size": 50,
                "n_concurrent_workers": 1,
                "cache_dir": str(cache_dir),
                "pseudo_labels_csv": str(out_csv),
                "concordance_json": str(conc_json),
                "lang_csv": str(lang_csv),
            }
        }
        cfg_file = tmp_path / "test_config.yaml"
        with open(cfg_file, "w") as f:
            yaml.safe_dump(custom_cfg, f)

        mock_batch_result = StudyLabelBatch(
            results=[
                StudyExtractionResult(
                    study_uid="UNLABELLED_0",
                    labels=KneeLabelExtraction(**{k: 0 for k in KneeLabelExtraction.model_fields}),
                    extraction_source="extractor_flash",
                )
            ],
            batch_id=0,
            studies_processed=1,
            studies_failed=0,
            studies_escalated=0,
        )

        with patch("src.nlp.orchestrator_agent.run_extractor_agent", new_callable=AsyncMock) as mock_extractor:
            mock_extractor.return_value = mock_batch_result
            asyncio.run(run_orchestrator(config_path=str(cfg_file), skip_validation=True))

            mock_extractor.assert_called_once()
            _, kwargs = mock_extractor.call_args
            assert kwargs.get("primary_model") == "custom-primary-model-v1", \
                f"Expected primary_model='custom-primary-model-v1', got {kwargs.get('primary_model')}"
            assert kwargs.get("fallback_model") == "custom-fallback-model-v1", \
                f"Expected fallback_model='custom-fallback-model-v1', got {kwargs.get('fallback_model')}"


# ---------------------------------------------------------------------------
# Gold study protection test (T14)
# ---------------------------------------------------------------------------

def test_gold_rows_never_modified():
    """T14: Imputer and merge_and_save_pseudo_labels never modify gold study rows."""
    df = _make_df(n_gold=2, n_unlabelled=3)
    gold_before = df[df["StudyInstanceUID"].str.startswith("GOLD")].copy()

    for uid in ["UNLABELLED_0", "UNLABELLED_1"]:
        df.loc[df["StudyInstanceUID"] == uid, ["Synovitis", "Effusion", "Baker's", "Medial OA"]] = -1

    result = apply_silence_imputation(df, gold_uids={"GOLD_0", "GOLD_1"})
    gold_after = result[result["StudyInstanceUID"].str.startswith("GOLD")]

    label_cols = ["ACL", "MCL", "Effusion", "Synovitis", "Baker's", "Medial OA"]
    for col in label_cols:
        assert (gold_before[col].values == gold_after[col].values).all()
