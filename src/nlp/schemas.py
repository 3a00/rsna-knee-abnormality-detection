"""
schemas.py -- Pydantic models for AGY structured output enforcement.

v3/v4 Design:
  StudyExtractionResult is defined BEFORE StudyLabelBatch (which references it).
  StudyLabelBatch.model_rebuild() is called at module bottom as belt-and-suspenders
  to ensure Pydantic v2 resolves all forward references regardless of import order.

These schemas are passed as `response_schema` to each Extractor subagent so the
AGY SDK validates the JSON at the protocol level -- no manual parsing needed.
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


# 3-state label: 1=positive, 0=explicit negative, -1=not addressed (silence)
LabelState = Literal[-1, 0, 1]


class KneeLabelExtraction(BaseModel):
    """Structured output schema for one study's 12-label extraction.

    Each field maps to one of the 12 RSNA target findings.
    Values must be exactly -1, 0, or 1 -- never None, never 2, never 0.5.
    The AGY SDK enforces this at the protocol level via response_schema.
    """
    acl_tear:          LabelState = Field(description="ACL: 1=high-grade/complete tear, 0=explicit negative, -1=not addressed")
    mcl_tear:          LabelState = Field(description="MCL: 1=high-grade/complete tear, 0=explicit negative, -1=not addressed")
    medial_meniscus:   LabelState = Field(description="Medial meniscus: 1=surface-contacting (2-slice rule), 0=explicit negative, -1=not addressed")
    lateral_meniscus:  LabelState = Field(description="Lateral meniscus: same 2-slice rule, 0=explicit negative, -1=not addressed")
    medial_oa:         LabelState = Field(description="Medial OA: 1=>1cm >50% loss, 0=explicit negative, -1=not addressed")
    lateral_oa:        LabelState = Field(description="Lateral OA: same threshold, 0=explicit negative, -1=not addressed")
    patellofemoral_oa: LabelState = Field(description="PF OA: same threshold, 0=explicit negative, -1=not addressed")
    joint_effusion:    LabelState = Field(description="Effusion: 1=moderate/large, 0=explicit negative, -1=not addressed")
    synovitis:         LabelState = Field(description="Synovitis: 1=marked thickening, 0=explicit negative, -1=not addressed")
    bakers_cyst:       LabelState = Field(description="Baker's cyst: 1=moderate/large, 0=explicit negative, -1=not addressed")
    bone_contusion:    LabelState = Field(description="Bone contusion: 1=traumatic marrow edema, 0=explicit negative, -1=not addressed")
    fracture:          LabelState = Field(description="Fracture: 1=acute cortical break, 0=explicit negative, -1=not addressed")


class StudyExtractionResult(BaseModel):
    """Single study extraction result bundled with its UID for merge."""
    study_uid: str
    labels: KneeLabelExtraction
    extraction_source: Literal["extractor_flash", "extractor_flash_fallback", "cache", "hard_fallback"]


class StudyLabelBatch(BaseModel):
    """Output schema for a batch of studies processed by one Extractor subagent."""
    results: list[StudyExtractionResult]
    batch_id: int
    studies_processed: int
    studies_failed: int          # Count that fell all the way to hard-fallback
    studies_escalated: int       # Count that needed Flash→gemini-3.6-flash escalation


class ConcordanceReport(BaseModel):
    """Output schema for the Validator subagent's concordance check."""
    macro_auc: float
    per_label_auc: dict[str, float]
    per_language_auc: dict[str, float]
    overall_ppv: float
    overall_recall: float
    silence_rates: dict[str, float]
    gold_studies_evaluated: int
    concordance_ceiling_note: str


StudyLabelBatch.model_rebuild()
