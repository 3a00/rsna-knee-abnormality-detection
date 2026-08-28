"""
imputer.py -- Post-extraction silence imputation for RSNA Phase 2.

Applies three empirically-validated rules from Discussion 01:
  1. Selective Synovitis imputation from Effusion (AUC +0.112)
  2. Structural absence zero-mapping (Baker's, Medial OA, Lateral OA)
  3. PF OA conservative zero-mapping

Called by the orchestrator agent after all extractor subagents finish,
and BEFORE saving pseudo_labels.csv.

Rules are NOT applied to the 58 gold study rows (those are authoritative 0/1).
"""

from __future__ import annotations

import pandas as pd


def apply_silence_imputation(
    df: pd.DataFrame,
    gold_uids: set[str] | None = None,
) -> pd.DataFrame:
    """Apply all three silence imputation rules to the extracted label table.

    This function modifies the DataFrame in-place and returns it.
    Rules apply ONLY to unaddressed silence (-1). Gold study labels (authoritative 0/1)
    are preserved intact.

    Args:
        df: DataFrame with at minimum columns:
            "Synovitis", "Effusion", "Baker's", "Medial OA",
            "Lateral OA", "PF OA", and all other 12 label columns.
            -1 = not addressed (silence), 0 = explicit negative, 1 = positive.
            NaN = extraction failed (kept as NaN for BCE loss masking).
        gold_uids: Optional set of StudyInstanceUIDs for gold studies to strictly protect.

    Returns:
        DataFrame with imputation rules applied. A new float column
        "synovitis_soft" is added for the soft synovitis label.
        A new boolean column "pf_oa_imputed" marks imputed PF OA cells.
    """
    LABEL_COLUMNS = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA",
        "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
    ]
    existing_label_cols = [c for c in LABEL_COLUMNS if c in df.columns]

    # Mask for unlabelled rows (or rows where UID is not in gold_uids)
    if gold_uids and "StudyInstanceUID" in df.columns:
        unlabelled_mask = ~df["StudyInstanceUID"].astype(str).isin(gold_uids)
    else:
        # Gold rows in RSNA have only 0 and 1; unlabelled rows contain -1 / NaN
        unlabelled_mask = pd.Series(True, index=df.index)

    # -----------------------------------------------------------------------
    # Rule 1: Synovitis Selective Imputation
    # P(syn | eff=1) = 0.63, P(syn | eff=0) = 0.22 (Discussion 01 empirical)
    # ONLY applied where synovitis == -1 (unaddressed silence)
    # NEVER overrides explicit 0 or 1
    # -----------------------------------------------------------------------
    if "synovitis_soft" not in df.columns:
        df["synovitis_soft"] = float("nan")

    syn_silence_mask = unlabelled_mask & (df["Synovitis"] == -1)
    eff_positive_mask = df["Effusion"] == 1

    # Where synovitis is silent AND effusion is positive → soft label 0.63
    df.loc[syn_silence_mask & eff_positive_mask, "synovitis_soft"] = 0.63
    # Where synovitis is silent AND effusion is NOT positive → soft label 0.22
    df.loc[syn_silence_mask & ~eff_positive_mask, "synovitis_soft"] = 0.22
    # Where synovitis is explicit (0 or 1), soft label mirrors it directly
    explicit_syn_mask = unlabelled_mask & (df["Synovitis"] != -1) & df["Synovitis"].notna()
    df.loc[explicit_syn_mask, "synovitis_soft"] = df.loc[explicit_syn_mask, "Synovitis"].astype(float)

    # -----------------------------------------------------------------------
    # Rule 2: Structural Absence Mapping
    # Baker's cyst, Medial OA, Lateral OA: silence = almost certainly absent
    # Gold+ rate when silent: 3%, 0%, 0% (Discussion 01)
    # -----------------------------------------------------------------------
    for col in ["Baker's", "Medial OA", "Lateral OA"]:
        if col in df.columns:
            silence_mask = unlabelled_mask & (df[col] == -1)
            df.loc[silence_mask, col] = 0

    # -----------------------------------------------------------------------
    # Rule 3: PF OA Conservative Zero-Mapping
    # Silence is "mixed" (21% gold+) but mapping 0 is conservative & stable
    # Track with a boolean flag for downstream sensitivity analysis
    # -----------------------------------------------------------------------
    if "pf_oa_imputed" not in df.columns:
        df["pf_oa_imputed"] = False

    if "PF OA" in df.columns:
        pf_silence_mask = unlabelled_mask & (df["PF OA"] == -1)
        df.loc[pf_silence_mask, "PF OA"] = 0
        df.loc[pf_silence_mask, "pf_oa_imputed"] = True

    # -----------------------------------------------------------------------
    # All other -1 values: keep as NaN (loss-masked during BCE training)
    # DO NOT call df.fillna(0) -- this invents false negatives
    # -----------------------------------------------------------------------
    remaining_silence_cols = [
        c for c in existing_label_cols
        if c not in ["Baker's", "Medial OA", "Lateral OA", "PF OA", "Synovitis"]
    ]
    for col in remaining_silence_cols:
        silence_mask = unlabelled_mask & (df[col] == -1)
        df.loc[silence_mask, col] = float("nan")

    return df
