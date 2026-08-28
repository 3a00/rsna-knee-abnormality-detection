"""
scan_dicom_fingerprints.py -- Phase 3 Step 0 (Kaggle only)

Scans one representative DICOM header per study to extract scanner fingerprint
metadata (Manufacturer, ManufacturerModelName, SoftwareVersions,
ImagingFrequency, ReceiveCoilName) via pydicom stop_before_pixels=True.

Writes data/labels/dicom_fingerprints.csv.

  train_series.csv does NOT contain scanner metadata -- it only has:
    StudyInstanceUID, SeriesInstanceUID, Fluid_Sensitive, Fat_Suppression,
    Anatomical_Plane.
    Scanner tags must be read from raw DICOM headers on Kaggle.

Run ONCE on Kaggle before building the CV split:
    python scripts/scan_dicom_fingerprints.py

Runtime: ~10-20 min for 4,407 studies (header-only reads are fast).
Output:  data/labels/dicom_fingerprints.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pydicom
import yaml


def scan_dicom_fingerprints(
    dicom_root: str,
    train_series_csv: str,
    output_csv: str,
    fingerprint_tags: list[str],
) -> pd.DataFrame:
    """Scan DICOM headers to extract scanner fingerprint metadata.

    Uses train_series.csv ONLY to enumerate study->series directory mappings.
    Reads actual scanner tags from the DICOM header via pydicom.

    Args:
        dicom_root: Path to DICOM root directory (Kaggle: /kaggle/input/...).
        train_series_csv: Path to train_series.csv (directory enumeration only).
        output_csv: Path to write dicom_fingerprints.csv.
        fingerprint_tags: DICOM attribute names from config.yaml cv.fingerprint_tags.

    Returns:
        DataFrame with StudyInstanceUID + one column per fingerprint tag
        + composite 'scanner_fingerprint' column.
    """
    dicom_root = Path(dicom_root)
    series_df = pd.read_csv(train_series_csv)

    # One representative series per study (first in CSV order)
    study_series = (
        series_df[["StudyInstanceUID", "SeriesInstanceUID"]]
        .drop_duplicates(subset="StudyInstanceUID")
        .reset_index(drop=True)
    )

    records = []
    n_studies = len(study_series)
    print(f"Scanning {n_studies} studies for scanner fingerprints...")

    for i, (_, row) in enumerate(study_series.iterrows()):
        study_uid  = row["StudyInstanceUID"]
        series_uid = row["SeriesInstanceUID"]

        # Support both dicom_root/train_series/{study}/{series} and dicom_root/{study}/{series}
        series_dir = dicom_root / "train_series" / study_uid / series_uid
        if not series_dir.exists():
            series_dir = dicom_root / study_uid / series_uid

        if i % 200 == 0:
            print(f"   [{i}/{n_studies}] Scanning...")

        record: dict = {"StudyInstanceUID": study_uid}

        try:
            # Try common DICOM extensions first
            dcm_files = sorted(series_dir.glob("*.dcm"))
            if not dcm_files:
                dcm_files = sorted(series_dir.glob("*.IMA"))
            if not dcm_files:
                dcm_files = [f for f in sorted(series_dir.iterdir()) if f.is_file()]

            if not dcm_files:
                raise FileNotFoundError(f"No DICOM files in {series_dir}")

            # stop_before_pixels=True: reads only header tags -- very fast
            ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)

            for tag_name in fingerprint_tags:
                try:
                    value = str(getattr(ds, tag_name, "UNKNOWN")).strip()
                    record[tag_name] = value if value else "UNKNOWN"
                except Exception:
                    record[tag_name] = "UNKNOWN"

        except Exception as e:
            warnings.warn(
                f"Could not read DICOM for study {study_uid}: {e}. "
                "All fingerprint tags set to UNKNOWN.",
                UserWarning,
                stacklevel=1,
            )
            for tag_name in fingerprint_tags:
                record[tag_name] = "UNKNOWN"

        records.append(record)

    df = pd.DataFrame(records)

    # Build composite scanner fingerprint string from all tags
    df["scanner_fingerprint"] = (
        df[fingerprint_tags]
        .fillna("UNKNOWN")
        .astype(str)
        .apply(lambda row: "|".join(row), axis=1)
    )

    n_unique = df["scanner_fingerprint"].nunique()
    all_unknown_fp = "|".join(["UNKNOWN"] * len(fingerprint_tags))
    n_unknown = (df["scanner_fingerprint"] == all_unknown_fp).sum()

    print(f"\n DICOM fingerprint scan complete:")
    print(f"   Studies scanned:        {len(df)}")
    print(f"   Unique fingerprints:    {n_unique}")
    print(f"   UNKNOWN studies:        {n_unknown}")
    print(f"   Top 10 fingerprints:")
    for fp, count in df["scanner_fingerprint"].value_counts().head(10).items():
        print(f"      {count:4d}x  {fp[:80]}")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n   Written: {output_csv}")

    return df


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    root = cfg["project"]["root"]

    scan_dicom_fingerprints(
        dicom_root=cfg["data"]["dicom_root"],
        train_series_csv=f"{root}/{cfg['data']['train_series_csv']}",
        output_csv=f"{root}/{cfg['cv']['fingerprints_csv']}",
        fingerprint_tags=cfg["cv"]["fingerprint_tags"],
    )
