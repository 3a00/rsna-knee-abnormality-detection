import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect, LangDetectException

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dicom_loader import (
    KneeDICOMLoader,
    SeriesType,
    AnatomicalPlane,
    normalize_mri_volume,
    load_and_normalize_series,
)
from src.utils.cv_splits import build_placeholder_split

config_path = PROJECT_ROOT / "config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

LABEL_COLS = cfg["labels"]["columns"]
print("=== RSNA Phase 1 Execution & Verification Pipeline ===")

# 1. Parse train.csv safely
train_path = PROJECT_ROOT / cfg["data"]["train_csv"]
df = pd.read_csv(train_path)

assert df.shape == (4407, 14), f"Unexpected shape: {df.shape}"
assert list(df.columns) == ["StudyInstanceUID", "Report"] + LABEL_COLS

gold_mask = df["ACL"].notna()
df_gold = df[gold_mask]
df_unlabelled = df[~gold_mask]

assert len(df_gold) == 58
assert len(df_unlabelled) == 4349
assert (df_unlabelled[LABEL_COLS] == 0).sum().sum() == 0

print(f"1. train.csv parsed cleanly: shape={df.shape}, gold={len(df_gold)}, unlabelled={len(df_unlabelled)}")

# 2. Language profiling
def detect_language(text: str) -> str:
    try:
        if pd.isna(text) or len(str(text).strip()) < 20:
            return "unknown"
        return detect(str(text))
    except LangDetectException:
        return "unknown"

print("2. Detecting report languages (4,407 studies)...")
df["detected_lang"] = df["Report"].apply(detect_language)
df_gold = df[df["ACL"].notna()]

print("   Top languages across all 4,407 reports:")
print(df["detected_lang"].value_counts().head(10).to_string())
print("\n   Language distribution in 58 gold studies:")
print(df_gold["detected_lang"].value_counts().to_string())

# 3. Gold label prevalence
prevalence = df_gold[LABEL_COLS].mean().sort_values(ascending=False)
mean_findings = df_gold[LABEL_COLS].sum(axis=1).mean()
print(f"\n3. Gold findings mean={mean_findings:.2f} per study.")
for label, pct in prevalence.items():
    print(f"   {label:25s}: {pct:.1%}")
assert 3.5 < mean_findings < 5.0

fig, ax = plt.subplots(figsize=(11, 5))
prevalence.plot(kind="bar", ax=ax, color="steelblue")
ax.set_title("Gold Study Positive Label Prevalence (n=58)", fontsize=13)
ax.set_ylabel("Fraction Positive")
ax.set_ylim(0, 1)
ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.4, label="50%")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

out_chart_path = PROJECT_ROOT / cfg["outputs"]["logs_dir"] / "gold_prevalence.png"
out_chart_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_chart_path, dpi=150)
plt.close()
print(f"   Prevalence chart saved: {out_chart_path}")

# 4. Normalization check
sample_vol = np.random.randint(0, 2000, size=(20, 256, 256))
norm_vol = normalize_mri_volume(
    sample_vol,
    percentile_low=cfg["normalization"]["percentile_low"],
    percentile_high=cfg["normalization"]["percentile_high"],
)
assert norm_vol.min() >= 0.0 and norm_vol.max() <= 1.0
assert norm_vol.dtype == np.float32
print("\n4. Volume-level normalization tested.")

# 5. Coverage check
train_series = pd.read_csv(PROJECT_ROOT / cfg["data"]["train_series_csv"])
test_series = pd.read_csv(PROJECT_ROOT / cfg["data"]["test_series_csv"])
print(f"\n5. train_series rows={len(train_series)}, unique studies={train_series['StudyInstanceUID'].nunique()}")
print(f"   Missing train plane: {train_series['Anatomical_Plane'].isna().sum()}, missing fluid: {train_series['Fluid_Sensitive'].isna().sum()}")
print(f"   test_series rows={len(test_series)}, unique studies={test_series['StudyInstanceUID'].nunique()}")
print(f"   Missing test plane: {test_series['Anatomical_Plane'].isna().sum()}, missing fluid: {test_series['Fluid_Sensitive'].isna().sum()}")

# 6. CV placeholder split
placeholder_csv_path = PROJECT_ROOT / cfg["cv"]["phase1_placeholder_csv"]
cv_df = build_placeholder_split(
    train_df=df,
    n_folds=cfg["cv"]["n_folds"],
    output_path=str(placeholder_csv_path),
    label_cols=LABEL_COLS,
    seed=cfg["project"]["seed"],
)
print(f"\n6. Placeholder CV split created at {placeholder_csv_path}")
print("   Overall fold sizes:", cv_df["fold_id"].value_counts().sort_index().to_dict())
print("   Gold studies per fold:", cv_df[cv_df["is_gold"]==1]["fold_id"].value_counts().sort_index().to_dict())
print("\n All Phase 1 steps executed and verified successfully!")
