"""
export_language_tags.py -- One-time script to persist Phase 1 language detection to disk.

Background:
  The Phase 1 EDA notebook (01_phase1_eda.ipynb) computes `detected_lang` for all 4,407
  studies but only prints the distribution -- it never saves it to disk. This means the
  Phase 2 Validator subagent has no language metadata to compute per-language AUC.

  This script re-runs the detection and saves:
    data/labels/language_tags.csv  (columns: StudyInstanceUID, detected_lang)

  The Validator reads this file. If it's missing, per-language AUC returns empty silently.
  Run this ONCE before running run_phase2_extraction.py.

Usage:
  python scripts/export_language_tags.py --config config.yaml
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def detect_language(text: str) -> str:
    """Detect language of a radiology report. Returns ISO 639-1 code or 'unknown'."""
    try:
        from langdetect import detect, LangDetectException
        if pd.isna(text) or len(str(text).strip()) < 20:
            return "unknown"
        return detect(str(text))
    except Exception:
        return "unknown"


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["project"]["root"])
    train_csv = root / cfg["data"]["train_csv"]
    output_path = root / cfg["nlp"]["lang_csv"]
    label_cols = cfg["labels"]["columns"]

    print(f"Loading {train_csv}...")
    df = pd.read_csv(train_csv, on_bad_lines="skip")

    print(f"Detecting language for {len(df)} reports...")
    df["detected_lang"] = df["Report"].apply(detect_language)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lang_df = df[["StudyInstanceUID", "detected_lang"]].copy()
    lang_df.to_csv(output_path, index=False)

    print(f"\n Saved {len(lang_df)} rows → {output_path}")
    print("\nFull dataset distribution:")
    print(lang_df["detected_lang"].value_counts().head(15).to_string())

    # Identify gold studies (any non-NaN in label_cols)
    gold_mask = df[label_cols].notna().any(axis=1)
    gold_langs = lang_df[gold_mask]["detected_lang"].value_counts()

    print(f"\nGold study distribution (n={gold_mask.sum()}):")
    print(gold_langs.to_string())
    print("\nExpected (Phase 1 authoritative): EN 28, ES 10, TR 6, HR 4, BG 3, EL 3, NL 2, DE 2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export language tags to disk for Phase 2 Validator")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
