"""
validator_agent.py -- AGY Validator Subagent for RSNA Phase 2.

Computes concordance metrics of extracted pseudo-labels vs. the 58 gold
studies. Runs ONCE after extraction completes -- do NOT tune against gold.

Output: ConcordanceReport (Pydantic) saved to outputs/logs/phase2_concordance.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .schemas import ConcordanceReport

logger = logging.getLogger(__name__)


# Gold language distribution (Phase 1 EDA authoritative):
# EN 28, ES 10, TR 6, HR 4, BG 3, EL 3, NL 2, DE 2
GOLD_LANGUAGE_DISTRIBUTION = {
    "en": 28, "es": 10, "tr": 6, "hr": 4,
    "bg": 3, "el": 3, "nl": 2, "de": 2,
}

LABEL_MAP = {
    "ACL": "acl_tear", "MCL": "mcl_tear",
    "Medial Meniscus": "medial_meniscus", "Lateral Meniscus": "lateral_meniscus",
    "Medial OA": "medial_oa", "Lateral OA": "lateral_oa",
    "PF OA": "patellofemoral_oa", "Effusion": "joint_effusion",
    "Synovitis": "synovitis", "Baker's": "bakers_cyst",
    "Contusion": "bone_contusion", "Fracture": "fracture",
}


def compute_concordance(
    pseudo_labels_csv: str,
    train_csv_path: str,
    lang_csv_path: str | None,
    output_json_path: str,
) -> ConcordanceReport:
    """Compute concordance metrics between pseudo-labels and 58 gold studies.

    This function is the sole entry point for Phase 2 validation.
    Call it ONCE after all extractions are complete.

    Args:
        pseudo_labels_csv: Path to data/labels/pseudo_labels.csv.
        train_csv_path: Path to data/raw/train.csv (contains gold labels).
        lang_csv_path: Optional path to CSV with study language tags.
        output_json_path: Where to save the concordance report JSON.

    Returns:
        ConcordanceReport Pydantic object.
    """
    label_cols = list(LABEL_MAP.keys())

    # Load pseudo labels (all 4,407 rows)
    pseudo_df = pd.read_csv(pseudo_labels_csv)

    # Load gold labels from train.csv
    gold_df = pd.read_csv(train_csv_path, on_bad_lines="skip")
    gold_mask = gold_df[label_cols].notna().any(axis=1)
    gold_df = gold_df[gold_mask].copy()

    logger.info(f"Gold studies for validation: {len(gold_df)} (expected 58)")

    # Align on StudyInstanceUID
    gold_uids = set(gold_df["StudyInstanceUID"].astype(str))
    pseudo_gold = pseudo_df[
        pseudo_df["StudyInstanceUID"].astype(str).isin(gold_uids)
    ].copy()

    # Per-label AUC (exclude -1 / NaN cells for that label)
    per_label_auc: dict[str, float] = {}
    all_gold_flat, all_pseudo_flat = [], []

    for csv_col in label_cols:
        gold_vals = gold_df.set_index("StudyInstanceUID")[csv_col]
        pseudo_vals = pseudo_gold.set_index("StudyInstanceUID")[csv_col]

        # Align on common index
        common_idx = gold_vals.index.intersection(pseudo_vals.index)
        g = gold_vals.loc[common_idx].values.astype(float)
        p = pseudo_vals.loc[common_idx].values.astype(float)

        # Exclude NaN and -1 in pseudo labels for this label
        valid_mask = (~np.isnan(g)) & (~np.isnan(p)) & (p != -1)
        g_valid, p_valid = g[valid_mask], p[valid_mask]

        if len(np.unique(g_valid)) < 2:
            logger.warning(f"Skipping AUC for {csv_col} -- only one class in gold subset")
            per_label_auc[csv_col] = float("nan")
            continue

        try:
            auc = roc_auc_score(g_valid, p_valid)
            per_label_auc[csv_col] = round(float(auc), 4)
            all_gold_flat.extend(g_valid.tolist())
            all_pseudo_flat.extend(p_valid.tolist())
        except Exception as e:
            logger.warning(f"AUC computation failed for {csv_col}: {e}")
            per_label_auc[csv_col] = float("nan")

    valid_aucs = [v for v in per_label_auc.values() if not np.isnan(v)]
    macro_auc = round(float(np.mean(valid_aucs)), 4) if valid_aucs else float("nan")

    # Per-language AUC (if language CSV is available)
    per_language_auc: dict[str, float] = {}
    if lang_csv_path and Path(lang_csv_path).exists():
        lang_df = pd.read_csv(lang_csv_path)
        for lang_code in GOLD_LANGUAGE_DISTRIBUTION:
            lang_uids = set(
                lang_df[lang_df["detected_lang"] == lang_code]["StudyInstanceUID"].astype(str)
            )
            lang_gold = gold_df[gold_df["StudyInstanceUID"].astype(str).isin(lang_uids)]
            lang_pseudo = pseudo_gold[pseudo_gold["StudyInstanceUID"].astype(str).isin(lang_uids)]

            if lang_gold.empty:
                continue

            lang_aucs = []
            for csv_col in label_cols:
                g = lang_gold[csv_col].values.astype(float)
                p_series = lang_pseudo.set_index("StudyInstanceUID")[csv_col]
                p = lang_gold["StudyInstanceUID"].map(
                    p_series.to_dict()
                ).values.astype(float)
                valid_mask = (~np.isnan(g)) & (~np.isnan(p)) & (p != -1)
                if valid_mask.sum() < 2 or len(np.unique(g[valid_mask])) < 2:
                    continue
                try:
                    lang_aucs.append(roc_auc_score(g[valid_mask], p[valid_mask]))
                except Exception:
                    pass

            if lang_aucs:
                per_language_auc[lang_code] = round(float(np.mean(lang_aucs)), 4)

    # Confusion metrics (flattened across all valid label-study pairs)
    g_arr = np.array(all_gold_flat)
    p_arr = np.array(all_pseudo_flat)
    pred_binary = (p_arr >= 1).astype(int)
    tp = int(((pred_binary == 1) & (g_arr == 1)).sum())
    fp = int(((pred_binary == 1) & (g_arr == 0)).sum())
    fn = int(((pred_binary == 0) & (g_arr == 1)).sum())
    tn = int(((pred_binary == 0) & (g_arr == 0)).sum())
    ppv = round(tp / (tp + fp), 4) if (tp + fp) > 0 else float("nan")
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else float("nan")

    # Silence rates per label (in pseudo labels for the 4,349 unlabelled rows)
    pseudo_unlabelled = pseudo_df[~pseudo_df["StudyInstanceUID"].astype(str).isin(gold_uids)]
    silence_rates: dict[str, float] = {}
    for csv_col in label_cols:
        if csv_col in pseudo_unlabelled.columns:
            n_total = len(pseudo_unlabelled)
            n_silence = (pseudo_unlabelled[csv_col] == -1).sum() + pseudo_unlabelled[csv_col].isna().sum()
            silence_rates[csv_col] = round(float(n_silence / n_total), 4) if n_total > 0 else float("nan")

    report = ConcordanceReport(
        macro_auc=macro_auc,
        per_label_auc=per_label_auc,
        per_language_auc=per_language_auc,
        overall_ppv=ppv,
        overall_recall=recall,
        silence_rates=silence_rates,
        gold_studies_evaluated=len(gold_df),
        concordance_ceiling_note=(
            "Structural ceiling ~82.5% (Discussion 05 empirical audit: 198/240 decisions). "
            "83-85% = best-in-class. Do NOT target 90%+ -- that is overfitting to gold labels."
        ),
    )

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json_path).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"Concordance report saved: {output_json_path}")
    logger.info(f"Macro AUC: {macro_auc:.4f} | PPV: {ppv:.4f} | Recall: {recall:.4f}")

    return report
