"""
nlp/ -- Phase 2: LLM-based weak label extraction from multilingual reports.

Public API:
    run_orchestrator()  -- Launch full extraction pipeline
    apply_silence_imputation()  -- Apply post-extraction silence rules
    compute_concordance()  -- Validate against 58 gold studies (once)
"""

try:
    from .orchestrator_agent import run_orchestrator
    from .validator_agent import compute_concordance
except ImportError:
    run_orchestrator = None
    compute_concordance = None

from .imputer import apply_silence_imputation
from .schemas import (
    KneeLabelExtraction,
    StudyLabelBatch,
    StudyExtractionResult,
    ConcordanceReport,
)

__all__ = [
    "run_orchestrator",
    "apply_silence_imputation",
    "compute_concordance",
    "KneeLabelExtraction",
    "StudyLabelBatch",
    "StudyExtractionResult",
    "ConcordanceReport",
]
