"""
dicom_loader.py -- DICOM Loader and MRI Series Classifier for RSNA Knee MRI

Primary routing priority:
  1. Pre-computed columns from train_series.csv / test_series.csv ("series_csv")
  2. EchoTime / ImageOrientationPatient DICOM physics tags ("physics")
  3. SeriesDescription text regex ("text_fallback")

Normalization: per-volume percentile clipping applied ONCE per series
(NOT per slice) to maintain consistent slice-to-slice contrast.

All paths passed as arguments -- never hardcoded.
Coverage of train_series.csv must be verified in Step 2.5 before
finalizing the primary/fallback role in Phase 3.
"""

from __future__ import annotations

import enum
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# =============================================================================
# Enums & Dataclasses
# =============================================================================

class SeriesType(enum.Enum):
    """MRI sequence type classification."""
    FLUID_SENSITIVE = "fluid_sensitive"   # T2-weighted, PD-FS, STIR
    ANATOMICAL      = "anatomical"        # T1-weighted, non-FS PD
    UNKNOWN         = "unknown"


class AnatomicalPlane(enum.Enum):
    """MRI imaging plane classification."""
    SAGITTAL = "sagittal"
    CORONAL  = "coronal"
    AXIAL    = "axial"
    UNKNOWN  = "unknown"


@dataclass
class DICOMSeries:
    """Container for one loaded MRI series and its classified metadata.

    Attributes:
        study_uid:             StudyInstanceUID of the parent study
        series_uid:            SeriesInstanceUID of this series
        series_type:           Classified sequence type
        plane:                 Classified anatomical plane
        pixel_array:           3D float32 array (n_slices, H, W), values in [0, 1].
                               Normalized with per-volume percentile clipping
                               (NOT fixed HU window). Computed over the full
                               volume -- NOT per-slice.
        echo_time:             TE in milliseconds (None if DICOM tag missing)
        repetition_time:       TR in milliseconds (None if DICOM tag missing)
        series_description:    Raw SeriesDescription string (informational only)
        n_slices:              Number of slices in the volume
        classification_source: "series_csv" | "physics" | "text_fallback" | "unknown"
    """
    study_uid:             str
    series_uid:            str
    series_type:           SeriesType
    plane:                 AnatomicalPlane
    pixel_array:           np.ndarray   # (n_slices, H, W), float32, [0, 1]
    echo_time:             Optional[float]
    repetition_time:       Optional[float]
    series_description:    str
    n_slices:              int
    classification_source: str


# =============================================================================
# Standalone Normalization and Series Loader (Single Source of Truth)
# =============================================================================

def normalize_mri_volume(
    pixel_array: np.ndarray,
    percentile_low: float = 0.5,
    percentile_high: float = 99.5,
    slope: float = 1.0,
    intercept: float = 0.0,
) -> np.ndarray:
    """Apply MRI-appropriate per-volume normalization.

    MRI intensities are NOT Hounsfield Units -- they are arbitrary and
    vendor-dependent. A fixed HU window (CT) will fail across multi-vendor
    datasets. This function clips to the [percentile_low, percentile_high]
    range of the ENTIRE 3D VOLUME, ensuring consistent slice-to-slice contrast.

    This is called ONCE per series (not once per slice) to prevent
    slice-to-slice contrast drift.

    Args:
        pixel_array: Raw stored int or float pixel data. Shape: (n_slices, H, W) or (H, W).
        percentile_low: Lower percentile for clipping (default 0.5)
        percentile_high: Upper percentile for clipping (default 99.5)
        slope: RescaleSlope DICOM tag value (default 1.0)
        intercept: RescaleIntercept DICOM tag value (default 0.0)

    Returns:
        Float32 array of same shape, values normalized in [0.0, 1.0].
    """
    arr = pixel_array.astype(np.float32) * slope + intercept
    lo = float(np.percentile(arr, percentile_low))
    hi = float(np.percentile(arr, percentile_high))
    if hi == lo:
        return np.zeros_like(arr)  # Blank/constant volume guard
    clipped = np.clip(arr, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def load_and_normalize_series(
    series_dir: Path | str,
    percentile_low: float = 0.5,
    percentile_high: float = 99.5,
) -> np.ndarray:
    """Load all DICOM slices in a series directory, stack into 3D volume,
    and apply volume-level MRI normalization.

    Args:
        series_dir: Directory containing all .dcm files for one series.
                    Files are sorted by InstanceNumber DICOM tag.
        percentile_low: Lower percentile for volume-level clipping
        percentile_high: Upper percentile for volume-level clipping

    Returns:
        Float32 array of shape (n_slices, H, W), values in [0.0, 1.0]

    Raises:
        FileNotFoundError: If series_dir does not exist or contains no .dcm files
        ValueError: If DICOM files have inconsistent spatial dimensions
    """
    import pydicom

    series_path = Path(series_dir)
    dcm_files = sorted(series_path.glob("*.dcm"))
    if not dcm_files:
        dcm_files = sorted(series_path.glob("*.IMA"))
    if not dcm_files:
        dcm_files = [f for f in sorted(series_path.iterdir()) if f.is_file()]
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found in {series_path}")

    # Load and sort slices by InstanceNumber (most reliable sort key)
    slices = []
    for path in dcm_files:
        ds = pydicom.dcmread(str(path))
        slices.append((int(getattr(ds, "InstanceNumber", 0)), ds))
    slices.sort(key=lambda x: x[0])

    # Stack raw pixels into a 3D array, applying per-file RescaleSlope/Intercept
    arrays = []
    for _, ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arrays.append(arr * slope + intercept)

    unique_shapes = set(a.shape for a in arrays)
    if len(unique_shapes) > 1:
        raise ValueError(
            f"Inconsistent slice dimensions in {series_path}: "
            f"{[a.shape for a in arrays]}"
        )

    volume = np.stack(arrays, axis=0)  # shape: (n_slices, H, W)

    # Normalize once over the full volume -- NOT per slice
    return normalize_mri_volume(volume, percentile_low=percentile_low, percentile_high=percentile_high)


# =============================================================================
# KneeDICOMLoader Class
# =============================================================================

class KneeDICOMLoader:
    """Loads and classifies knee MRI DICOM series for RSNA competition.

    Classification priority:
      1. Pre-computed columns from train_series.csv / test_series.csv ("series_csv")
      2. Physics tags: EchoTime (sequence type), ImageOrientationPatient (plane) ("physics")
      3. SeriesDescription text regex ("text_fallback")

    Normalization: per-volume percentile clipping (0.5-99.5th percentile by default),
    applied ONCE per series after stacking all slices. Never applied per-slice.

    Args:
        dicom_root:            Root directory of competition DICOM data
        te_fluid_threshold_ms: EchoTime threshold for fluid-sensitive classification.
                               Default 30ms. Used only when series_csv is unavailable.
        percentile_low:        Lower percentile for volume normalization (default 0.5)
        percentile_high:       Upper percentile for volume normalization (default 99.5)
    """

    _FLUID_SENSITIVE_RE = re.compile(
        r"(T2|PD[\s_]?FS|PD[\s_]?FAT|STIR|FATSAT|fat[\s_]?sat|"
        r"fs|Dixon_W|MEDIC|DESS)",
        re.IGNORECASE,
    )
    _ANATOMICAL_RE = re.compile(
        r"\b(T1|PD\b(?![\s_]?FS)|non[\s_]?FS|anatomical)\b",
        re.IGNORECASE,
    )
    _PLANE_THRESHOLD = 0.8  # Minimum |component| for axis alignment via IOP

    def __init__(
        self,
        dicom_root: str | Path,
        te_fluid_threshold_ms: float = 30.0,
        percentile_low:  float = 0.5,
        percentile_high: float = 99.5,
    ) -> None:
        self.dicom_root = Path(dicom_root)
        self.te_fluid_threshold_ms = te_fluid_threshold_ms
        self.percentile_low  = percentile_low
        self.percentile_high = percentile_high

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def load_study(self, study_uid: str) -> list[DICOMSeries]:
        """Load all series for one study and return classified DICOMSeries objects.

        Args:
            study_uid: StudyInstanceUID string

        Returns:
            List of classified DICOMSeries objects.
        """
        study_dir = self.dicom_root / study_uid
        if not study_dir.exists() and (self.dicom_root / "train_series" / study_uid).exists():
            study_dir = self.dicom_root / "train_series" / study_uid
        return load_study_series(
            study_dir=study_dir,
            percentile_low=self.percentile_low,
            percentile_high=self.percentile_high,
        )

    # ------------------------------------------------------------------
    # Classification methods
    # ------------------------------------------------------------------

    def classify_series_type(
        self,
        echo_time_ms:               Optional[float],
        repetition_time_ms:         Optional[float],
        series_description:         str,
        series_csv_fluid_sensitive: Optional[int] = None,
    ) -> tuple[SeriesType, str]:
        """Classify MRI sequence type using 3-priority fallback chain.

        Priority:
          1. series_csv_fluid_sensitive (int 0/1) -> "series_csv"
          2. EchoTime > te_fluid_threshold_ms     -> "physics"
          3. SeriesDescription regex match        -> "text_fallback"

        Args:
            echo_time_ms:               EchoTime in ms (None if tag missing)
            repetition_time_ms:         RepetitionTime in ms (None if missing)
            series_description:         Raw SeriesDescription string
            series_csv_fluid_sensitive: Pre-computed 0/1 from series CSV
                                        (None if series not in CSV)

        Returns:
            Tuple of (SeriesType enum, classification_source string)
        """
        # Priority 1: series_csv
        if series_csv_fluid_sensitive is not None:
            t = (SeriesType.FLUID_SENSITIVE if series_csv_fluid_sensitive == 1
                 else SeriesType.ANATOMICAL)
            return t, "series_csv"

        # Priority 2: Physics (EchoTime)
        if echo_time_ms is not None:
            return (
                (SeriesType.FLUID_SENSITIVE, "physics")
                if echo_time_ms > self.te_fluid_threshold_ms
                else (SeriesType.ANATOMICAL, "physics")
            )

        # Priority 3: Text regex fallback
        if series_description:
            if self._FLUID_SENSITIVE_RE.search(series_description):
                return SeriesType.FLUID_SENSITIVE, "text_fallback"
            if self._ANATOMICAL_RE.search(series_description):
                return SeriesType.ANATOMICAL, "text_fallback"

        return SeriesType.UNKNOWN, "unknown"

    def classify_plane(
        self,
        image_orientation_patient: Optional[list[float]],
        series_description:        str,
        series_csv_plane:          Optional[str] = None,
    ) -> tuple[AnatomicalPlane, str]:
        """Classify MRI imaging plane using 3-priority fallback chain.

        Priority:
          1. series_csv_plane string (e.g. "Sagittal") -> "series_csv"
          2. ImageOrientationPatient cross-product       -> "physics"
          3. SeriesDescription text keywords             -> "text_fallback"

        Args:
            image_orientation_patient: 6-element list from DICOM IOP tag
                                       (None if tag missing)
            series_description:        Raw SeriesDescription string
            series_csv_plane:          Pre-computed plane string from series CSV

        Returns:
            Tuple of (AnatomicalPlane enum, classification_source string)
        """
        _PLANE_STR_MAP = {
            "sagittal": AnatomicalPlane.SAGITTAL,
            "coronal":  AnatomicalPlane.CORONAL,
            "axial":    AnatomicalPlane.AXIAL,
        }

        # Priority 1: series_csv
        if series_csv_plane is not None:
            plane = _PLANE_STR_MAP.get(series_csv_plane.lower(), AnatomicalPlane.UNKNOWN)
            return plane, "series_csv"

        # Priority 2: Physics (ImageOrientationPatient)
        if image_orientation_patient and len(image_orientation_patient) == 6:
            row_vec = np.array(image_orientation_patient[:3])
            col_vec = np.array(image_orientation_patient[3:])
            normal  = np.cross(row_vec, col_vec)
            abs_n   = np.abs(normal)
            dom     = int(np.argmax(abs_n))
            if abs_n[dom] > self._PLANE_THRESHOLD:
                iop_map = {0: AnatomicalPlane.SAGITTAL,
                           1: AnatomicalPlane.CORONAL,
                           2: AnatomicalPlane.AXIAL}
                return iop_map[dom], "physics"

        # Priority 3: Text fallback
        desc = series_description.lower()
        if any(k in desc for k in ["sag", "sagital", "sagittal"]):
            return AnatomicalPlane.SAGITTAL, "text_fallback"
        if any(k in desc for k in ["cor", "coronal", "frontal"]):
            return AnatomicalPlane.CORONAL, "text_fallback"
        if any(k in desc for k in ["ax", "axial", "tra", "transverse"]):
            return AnatomicalPlane.AXIAL, "text_fallback"

        return AnatomicalPlane.UNKNOWN, "unknown"

    # ------------------------------------------------------------------
    # Normalization (delegates to single source of truth)
    # ------------------------------------------------------------------

    def normalize_mri_volume(
        self,
        pixel_array: np.ndarray,
        slope:       float = 1.0,
        intercept:   float = 0.0,
    ) -> np.ndarray:
        """Apply MRI-appropriate per-volume normalization using instance percentiles."""
        return normalize_mri_volume(
            pixel_array=pixel_array,
            percentile_low=self.percentile_low,
            percentile_high=self.percentile_high,
            slope=slope,
            intercept=intercept,
        )


# =============================================================================
# Standalone load_study_series Function
# =============================================================================

def load_study_series(
    study_dir: Path | str,
    series_metadata: Optional[any] = None,
    percentile_low: float = 0.5,
    percentile_high: float = 99.5,
) -> list[DICOMSeries]:
    """Load all DICOM series in a study directory and return classified DICOMSeries list.

    Args:
        study_dir: Directory containing series subdirectories for one study.
        series_metadata: Optional DataFrame from train_series.csv for metadata lookup.
        percentile_low: Lower percentile for volume normalization.
        percentile_high: Upper percentile for volume normalization.

    Returns:
        List of DICOMSeries dataclass instances.
    """
    import pydicom

    study_path = Path(study_dir)
    study_uid = study_path.name
    if not study_path.exists():
        raise FileNotFoundError(f"Study directory not found: {study_path}")

    series_dirs = [d for d in sorted(study_path.iterdir()) if d.is_dir()]
    if not series_dirs:
        # Check if study_dir itself contains .dcm or .IMA files
        dcm_check = list(study_path.glob("*.dcm")) + list(study_path.glob("*.IMA"))
        if dcm_check:
            series_dirs = [study_path]

    classifier = KneeDICOMLoader(
        study_path.parent,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )
    series_list: list[DICOMSeries] = []

    for s_dir in series_dirs:
        series_uid = s_dir.name
        dcm_files = sorted(s_dir.glob("*.dcm"))
        if not dcm_files:
            dcm_files = sorted(s_dir.glob("*.IMA"))
        if not dcm_files:
            dcm_files = [f for f in sorted(s_dir.iterdir()) if f.is_file()]
        if not dcm_files:
            continue

        try:
            ds_head = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
            te = (
                float(getattr(ds_head, "EchoTime", 0.0))
                if hasattr(ds_head, "EchoTime") and ds_head.EchoTime != ""
                else None
            )
            tr = (
                float(getattr(ds_head, "RepetitionTime", 0.0))
                if hasattr(ds_head, "RepetitionTime") and ds_head.RepetitionTime != ""
                else None
            )
            desc = str(getattr(ds_head, "SeriesDescription", "") or "")
            iop_raw = getattr(ds_head, "ImageOrientationPatient", None)
            iop = (
                [float(x) for x in iop_raw]
                if iop_raw is not None and len(iop_raw) == 6
                else None
            )

            pixel_array = load_and_normalize_series(
                s_dir,
                percentile_low=percentile_low,
                percentile_high=percentile_high,
            )
            n_slices = pixel_array.shape[0]

            csv_fluid = None
            csv_plane = None
            if series_metadata is not None and hasattr(series_metadata, "empty") and not series_metadata.empty:
                match = series_metadata[
                    (series_metadata["StudyInstanceUID"] == study_uid) &
                    (series_metadata["SeriesInstanceUID"] == series_uid)
                ]
                if not match.empty:
                    if "Fluid_Sensitive" in match.columns and match.iloc[0]["Fluid_Sensitive"] is not None:
                        csv_fluid = int(match.iloc[0]["Fluid_Sensitive"])
                    if "Anatomical_Plane" in match.columns and match.iloc[0]["Anatomical_Plane"] is not None:
                        csv_plane = str(match.iloc[0]["Anatomical_Plane"])

            stype, _ = classifier.classify_series_type(
                te, tr, desc, series_csv_fluid_sensitive=csv_fluid
            )
            plane, src = classifier.classify_plane(
                iop, desc, series_csv_plane=csv_plane
            )

            series_list.append(
                DICOMSeries(
                    study_uid=study_uid,
                    series_uid=series_uid,
                    series_type=stype,
                    plane=plane,
                    pixel_array=pixel_array,
                    echo_time=te,
                    repetition_time=tr,
                    series_description=desc,
                    n_slices=n_slices,
                    classification_source=src,
                )
            )
        except Exception as e:
            warnings.warn(
                f"Failed to load series {series_uid} in {study_uid}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

    return series_list
