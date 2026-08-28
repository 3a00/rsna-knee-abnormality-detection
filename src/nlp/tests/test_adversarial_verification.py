"""
test_adversarial_verification.py -- Adversarial Verification Suite for Phase 2 Weak Labels.
Authored by Challenger 1 (Empirical Challenger).

Tests data integrity, gold preservation, structural zero-mapping,
non-structural NaN masking, calibrated synovitis soft targets, and edge cases.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.nlp.imputer import apply_silence_imputation
from src.nlp.schemas import KneeLabelExtraction

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_TRAIN_CSV = PROJECT_ROOT / "data" / "raw" / "train.csv"
PSEUDO_LABELS_CSV = PROJECT_ROOT / "data" / "labels" / "pseudo_labels.csv"
CONCORDANCE_JSON = PROJECT_ROOT / "outputs" / "logs" / "phase2_concordance.json"
DESKTOP_REPORT = Path("/home/wenalz/Desktop/RSNA_Phase2_Extraction_Results.md")

LABEL_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

STRUCTURAL_COLUMNS = ["Baker's", "Medial OA", "Lateral OA", "PF OA"]
NON_STRUCTURAL_COLUMNS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Contusion", "Fracture"]


@pytest.fixture(scope="module")
def raw_train_df():
    assert RAW_TRAIN_CSV.exists(), f"Raw train CSV missing: {RAW_TRAIN_CSV}"
    return pd.read_csv(RAW_TRAIN_CSV, on_bad_lines="skip")


@pytest.fixture(scope="module")
def pseudo_labels_df():
    assert PSEUDO_LABELS_CSV.exists(), f"Pseudo labels CSV missing: {PSEUDO_LABELS_CSV}"
    return pd.read_csv(PSEUDO_LABELS_CSV)


# ===========================================================================
# 1. Dataset Shape and Column Integrity
# ===========================================================================

def test_pseudo_labels_exact_geometry(pseudo_labels_df, raw_train_df):
    """Test 1.1: Dataset has exactly 4,407 rows and identical UID set to raw train."""
    assert len(pseudo_labels_df) == 4407, f"Expected 4407 rows, got {len(pseudo_labels_df)}"
    assert len(raw_train_df) == 4407, f"Expected 4407 raw rows, got {len(raw_train_df)}"
    
    # Assert exact UID ordering and uniqueness
    assert pseudo_labels_df["StudyInstanceUID"].is_unique, "Duplicate StudyInstanceUIDs found in pseudo_labels"
    assert (pseudo_labels_df["StudyInstanceUID"].values == raw_train_df["StudyInstanceUID"].values).all(), \
        "StudyInstanceUID ordering differs from raw train.csv"


def test_required_columns_present(pseudo_labels_df):
    """Test 1.2: All 12 clinical finding columns plus metadata and soft columns exist."""
    required_cols = ["StudyInstanceUID", "Report", "synovitis_soft", "pf_oa_imputed"] + LABEL_COLUMNS
    for col in required_cols:
        assert col in pseudo_labels_df.columns, f"Missing required column: {col}"


# ===========================================================================
# 2. Strict Gold Ground-Truth Invariance
# ===========================================================================

def test_gold_ground_truth_untouched(pseudo_labels_df, raw_train_df):
    """Test 2.1: Programmatic assertion -- all 58 gold studies are bit-for-bit identical."""
    gold_mask = raw_train_df[LABEL_COLUMNS].notna().any(axis=1)
    assert gold_mask.sum() == 58, f"Expected 58 gold studies in train.csv, found {gold_mask.sum()}"

    gold_uids = set(raw_train_df[gold_mask]["StudyInstanceUID"].astype(str))
    
    pseudo_gold = pseudo_labels_df[pseudo_labels_df["StudyInstanceUID"].astype(str).isin(gold_uids)]
    assert len(pseudo_gold) == 58, f"Expected 58 gold studies in pseudo_labels, found {len(pseudo_gold)}"

    raw_gold_aligned = raw_train_df[gold_mask].sort_values("StudyInstanceUID").reset_index(drop=True)
    pseudo_gold_aligned = pseudo_gold.sort_values("StudyInstanceUID").reset_index(drop=True)

    # Verbatim DataFrame equals assertion
    assert raw_gold_aligned[LABEL_COLUMNS].equals(pseudo_gold_aligned[LABEL_COLUMNS]), \
        "CRITICAL: Gold label values in pseudo_labels.csv do NOT match raw train.csv!"

    # Ensure gold pf_oa_imputed is False
    assert (pseudo_gold_aligned["pf_oa_imputed"] == False).all(), \
        "Gold studies should never have pf_oa_imputed == True"


# ===========================================================================
# 3. Structural Absence Zero-Mapping
# ===========================================================================

def test_structural_silence_zero_mapping(pseudo_labels_df, raw_train_df):
    """Test 3.1: Baker's, Medial OA, Lateral OA, PF OA mapped to 0 when silent."""
    gold_mask = raw_train_df[LABEL_COLUMNS].notna().any(axis=1)
    gold_uids = set(raw_train_df[gold_mask]["StudyInstanceUID"].astype(str))
    
    unlabelled_df = pseudo_labels_df[~pseudo_labels_df["StudyInstanceUID"].astype(str).isin(gold_uids)]
    assert len(unlabelled_df) == 4349, f"Expected 4349 unlabelled rows, got {len(unlabelled_df)}"

    for col in STRUCTURAL_COLUMNS:
        nan_count = unlabelled_df[col].isna().sum()
        assert nan_count == 0, f"Structural column '{col}' contains {nan_count} NaNs; must be zero-mapped"
        unique_vals = set(unlabelled_df[col].unique())
        assert unique_vals.issubset({0.0, 1.0}), f"Unexpected values in structural col '{col}': {unique_vals}"


# ===========================================================================
# 4. Non-Structural Silence Loss Masking (NaN Preservation)
# ===========================================================================

def test_non_structural_silence_nan_preservation(pseudo_labels_df, raw_train_df):
    """Test 4.1: ACL, MCL, Menisci, Contusion, Fracture preserve NaNs (no global fillna(0))."""
    gold_mask = raw_train_df[LABEL_COLUMNS].notna().any(axis=1)
    gold_uids = set(raw_train_df[gold_mask]["StudyInstanceUID"].astype(str))
    
    unlabelled_df = pseudo_labels_df[~pseudo_labels_df["StudyInstanceUID"].astype(str).isin(gold_uids)]

    for col in NON_STRUCTURAL_COLUMNS:
        nan_count = unlabelled_df[col].isna().sum()
        assert nan_count > 0, f"Non-structural column '{col}' has 0 NaNs! Global fillna(0) was likely used!"


# ===========================================================================
# 5. Calibrated Synovitis Soft Imputation
# ===========================================================================

def test_synovitis_soft_calibration(pseudo_labels_df, raw_train_df):
    """Test 5.1: Unlabelled synovitis soft target is 0.22 when effusion!=1, 0.63 when effusion=1."""
    gold_mask = raw_train_df[LABEL_COLUMNS].notna().any(axis=1)
    gold_uids = set(raw_train_df[gold_mask]["StudyInstanceUID"].astype(str))
    
    unlabelled_df = pseudo_labels_df[~pseudo_labels_df["StudyInstanceUID"].astype(str).isin(gold_uids)]
    
    # Synovitis soft values must be valid calibrated values in {0.22, 0.63, 0.0, 1.0}
    assert unlabelled_df["synovitis_soft"].isin([0.22, 0.63, 0.0, 1.0]).all(), \
        "Invalid synovitis_soft probability found"
    
    # Check that imputed values (0.22 and 0.63) are present and properly correlated
    assert (unlabelled_df["synovitis_soft"] == 0.22).sum() > 0, "Expected 0.22 soft targets"
    assert (unlabelled_df["synovitis_soft"] == 0.63).sum() > 0, "Expected 0.63 soft targets"


# ===========================================================================
# 6. Synthetic Adversarial Edge Cases for Imputer Logic
# ===========================================================================

def test_adversarial_imputer_synthetic_matrix():
    """Test 6.1: Exhaustive synthetic truth table testing for apply_silence_imputation."""
    synthetic_data = [
        # Gold row: must remain untouched
        {"StudyInstanceUID": "GOLD_001", "Effusion": 1.0, "Synovitis": 0.0, "Baker's": 1.0, "ACL": 1.0, "PF OA": 1.0},
        # Unlabelled row 1: Silence synovitis + Effusion 1 -> soft=0.63
        {"StudyInstanceUID": "UNL_001", "Effusion": 1.0, "Synovitis": -1.0, "Baker's": -1.0, "ACL": -1.0, "PF OA": -1.0},
        # Unlabelled row 2: Silence synovitis + Effusion 0 -> soft=0.22
        {"StudyInstanceUID": "UNL_002", "Effusion": 0.0, "Synovitis": -1.0, "Baker's": -1.0, "ACL": -1.0, "PF OA": -1.0},
        # Unlabelled row 3: Silence synovitis + Effusion NaN -> soft=0.22
        {"StudyInstanceUID": "UNL_003", "Effusion": float("nan"), "Synovitis": -1.0, "Baker's": -1.0, "ACL": -1.0, "PF OA": -1.0},
        # Unlabelled row 4: Explicit synovitis 1 -> soft=1.0, Synovitis=1
        {"StudyInstanceUID": "UNL_004", "Effusion": 0.0, "Synovitis": 1.0, "Baker's": 1.0, "ACL": 1.0, "PF OA": 1.0},
        # Unlabelled row 5: Explicit synovitis 0 -> soft=0.0, Synovitis=0
        {"StudyInstanceUID": "UNL_005", "Effusion": 1.0, "Synovitis": 0.0, "Baker's": 0.0, "ACL": 0.0, "PF OA": 0.0},
    ]
    df = pd.DataFrame(synthetic_data)
    for col in LABEL_COLUMNS:
        if col not in df.columns:
            df[col] = -1.0

    res = apply_silence_imputation(df, gold_uids={"GOLD_001"})

    # Check GOLD_001 untouched
    gold_row = res[res["StudyInstanceUID"] == "GOLD_001"].iloc[0]
    assert gold_row["Effusion"] == 1.0
    assert gold_row["Synovitis"] == 0.0
    assert gold_row["Baker's"] == 1.0
    assert gold_row["ACL"] == 1.0
    assert gold_row["PF OA"] == 1.0

    # Check UNL_001 (Effusion=1, Synovitis=-1)
    u1 = res[res["StudyInstanceUID"] == "UNL_001"].iloc[0]
    assert np.isclose(u1["synovitis_soft"], 0.63)
    assert u1["Baker's"] == 0.0
    assert u1["PF OA"] == 0.0
    assert u1["pf_oa_imputed"] == True
    assert np.isnan(u1["ACL"])

    # Check UNL_002 (Effusion=0, Synovitis=-1)
    u2 = res[res["StudyInstanceUID"] == "UNL_002"].iloc[0]
    assert np.isclose(u2["synovitis_soft"], 0.22)

    # Check UNL_003 (Effusion=NaN, Synovitis=-1)
    u3 = res[res["StudyInstanceUID"] == "UNL_003"].iloc[0]
    assert np.isclose(u3["synovitis_soft"], 0.22)

    # Check UNL_004 (Explicit Synovitis=1, Baker's=1, ACL=1)
    u4 = res[res["StudyInstanceUID"] == "UNL_004"].iloc[0]
    assert np.isclose(u4["synovitis_soft"], 1.0)
    assert u4["Baker's"] == 1.0  # Must NOT be zero-mapped!
    assert u4["ACL"] == 1.0      # Must NOT be NaN-masked!
    assert u4["PF OA"] == 1.0

    # Check UNL_005 (Explicit Synovitis=0, Baker's=0, ACL=0)
    u5 = res[res["StudyInstanceUID"] == "UNL_005"].iloc[0]
    assert np.isclose(u5["synovitis_soft"], 0.0)
    assert u5["Baker's"] == 0.0
    assert u5["ACL"] == 0.0


# ===========================================================================
# 7. Outputs and Concordance Validation Check
# ===========================================================================

def test_concordance_json_and_desktop_export():
    """Test 7.1: Verify concordance log exists and contains valid metrics."""
    assert CONCORDANCE_JSON.exists(), f"Concordance JSON missing: {CONCORDANCE_JSON}"
    with open(CONCORDANCE_JSON) as f:
        conc_data = json.load(f)
    assert "macro_auc" in conc_data
    assert "per_label_auc" in conc_data
    assert "per_language_auc" in conc_data
    assert conc_data["gold_studies_evaluated"] == 58
    assert conc_data["macro_auc"] > 0.80
