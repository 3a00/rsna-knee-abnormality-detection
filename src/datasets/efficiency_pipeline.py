"""
src/datasets/efficiency_pipeline.py

Production DICOM Preprocessing Pipeline for RSNA Knee MRI Efficiency Track.
Reconstructs true 3D spatial slice ordering via sign-canonical physical normal projection (k = p . n),
extracts fixed 140 mm physical FOV crops normalized by PixelSpacing (with zero-padding for small FOVs),
routes series into 6 standardized anatomical contrast slots, and forms 2.5D adjacent slice triplets.

Only 18 DICOM pixel arrays are decoded per study.
"""

from __future__ import annotations

import concurrent.futures
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset

# Prevent OpenCV internal threads from competing with DataLoader / ThreadPool workers
cv2.setNumThreads(0)

SLOT_NAMES: tuple[str, ...] = (
    "SAG_FLUID_FS",    # Slot 0
    "COR_FLUID_FS",    # Slot 1
    "AX_FLUID_FS",     # Slot 2
    "SAG_FLUID_NOFS",  # Slot 3
    "COR_T1",          # Slot 4
    "SAG_T1",          # Slot 5
)
NUM_SLOTS = 6
DEFAULT_PIXEL_SPACING = (0.4, 0.4)
DEFAULT_CROP_MM = 140.0
BUDGET_S = 0.50
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

FAST_HEADER_TAGS = [
    "SeriesInstanceUID",
    "InstanceNumber",
    "SeriesDescription",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "PixelSpacing",
    "EchoTime",
    "EchoNumbers",
    "RepetitionTime",
    "ImageType",
]

_FS_REGEX = re.compile(
    r"(?:\b|_)(PDW?[-_]?FS|FAT[-_\s]?(?:SAT|SUPP?)|FATSAT|STIR|TIRM|SPAIR|SPIR|CHESS|WATER|DIXON[-_]?W|FS)(?:\b|_)",
    re.IGNORECASE,
)
_NON_FS_PREFIX_REGEX = re.compile(
    r"(?:\b|_)(?:NON|NO|WITHOUT|W/O)[-_ ]?(?:FS|FAT[-_\s]?(?:SAT|SUPP?)|FATSAT)(?:\b|_)",
    re.IGNORECASE,
)
_T1_REGEX = re.compile(r"(?:\b|_)(T1W?)(?:\b|_)", re.IGNORECASE)
_PD_REGEX = re.compile(r"(?:\b|_)(PDW?|PROTON)(?:\b|_)", re.IGNORECASE)
_FLUID_REGEX = re.compile(
    r"(?:\b|_)(T2W?|STIR|TIRM|SPAIR|SPIR|DIXON|MEDIC|DESS|PDW?[-_]?FS)(?:\b|_)",
    re.IGNORECASE,
)
_QUANT_MAP_REGEX = re.compile(r"(?:\b|_)(MAP|MAPPING|T1RHO|T2_?MAP)(?:\b|_)", re.IGNORECASE)
_SOFT_EXCLUDE_REGEX = re.compile(
    r"(?:\b|_)(LOCALIZER|SCOUT|LOC|SURVEY|CALIBRATION|SCREEN_?SAVE)(?:\b|_)",
    re.IGNORECASE,
)
_PLANE_TEXT_REGEX = {
    "sagittal": re.compile(r"(?:\b|_)(SAG|SAGITTAL)(?:\b|_)", re.IGNORECASE),
    "coronal":  re.compile(r"(?:\b|_)(COR|CORONAL|FRONTAL)(?:\b|_)", re.IGNORECASE),
    "axial":    re.compile(r"(?:\b|_)(AX|AXIAL|TRA|TRANSVERSE)(?:\b|_)", re.IGNORECASE),
}

_TRUE_VALUES = {"1", "true", "yes", "y", "t"}
_FALSE_VALUES = {"0", "false", "no", "n", "f"}


def _to_flag(v: Any) -> Optional[int]:
    """Return 1/0, or None when unknown (NaN, empty, unparseable)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().lower()
    if s in _TRUE_VALUES:
        return 1
    if s in _FALSE_VALUES:
        return 0
    try:
        return int(float(s) != 0.0)
    except ValueError:
        return None


def normalize_plane_str(p: str | Any) -> str:
    """Normalize plane strings from metadata CSVs or text headers."""
    if p is None:
        return "unknown"
    p_clean = str(p).strip().lower()
    if "sag" in p_clean:
        return "sagittal"
    if "cor" in p_clean:
        return "coronal"
    if "ax" in p_clean or "tra" in p_clean:
        return "axial"
    return "unknown"


def load_series_metadata(
    csv_path: str | Path,
) -> Optional[dict[str, tuple[Optional[int], Optional[int], str]]]:
    """Load Fluid_Sensitive / Fat_Suppression / Anatomical_Plane; unknown values stay None."""
    p = Path(csv_path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    req = {"SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"}
    if not req.issubset(df.columns):
        warnings.warn(f"{p} missing columns {req - set(df.columns)}; using DICOM heuristics.", UserWarning)
        return None
    return {
        str(u): (_to_flag(fl), _to_flag(fs), normalize_plane_str(pl))
        for u, fl, fs, pl in zip(
            df["SeriesInstanceUID"], df["Fluid_Sensitive"], df["Fat_Suppression"], df["Anatomical_Plane"]
        )
    }


@dataclass(slots=True)
class DICOMHeaderMeta:
    """Metadata extracted during fast header scan."""
    path: str
    series_uid: str
    k_pos: float
    has_physical_coords: bool
    plane: str
    is_fluid: bool
    is_fs: bool
    pixel_spacing: tuple[float, float]
    instance_number: int
    echo_number: int = 1


def compute_slice_normal_projection(
    iop: Sequence[float],
    ipp: Sequence[float],
) -> tuple[float, str]:
    """Compute canonical physical normal coordinate k = p . n and anatomical plane.
    
    Canonicalization: n is oriented such that its dominant component is strictly positive.
    In DICOM patient LPS coordinates, canonical k strictly increases along:
      - Sagittal (X dominant): Right -> Left (+X).
      - Coronal  (Y dominant): Anterior -> Posterior (+Y).
      - Axial    (Z dominant): Inferior -> Superior (+Z).
    """
    r = np.array(iop[:3], dtype=np.float64)
    c = np.array(iop[3:], dtype=np.float64)
    n = np.cross(r, c)
    norm_n = np.linalg.norm(n)
    if norm_n > 1e-6:
        n = n / norm_n
    else:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    abs_n = np.abs(n)
    dom = int(np.argmax(abs_n))
    if n[dom] < 0.0:
        n = -n

    p = np.array(ipp, dtype=np.float64)
    k = float(np.dot(p, n))

    plane_map = {0: "sagittal", 1: "coronal", 2: "axial"}
    plane = plane_map.get(dom, "unknown")

    return k, plane


def parse_dicom_header_fast(
    file_path: str | Path,
    series_metadata: Optional[dict[str, tuple[Optional[int], Optional[int], str]]] = None,
) -> tuple[Optional[DICOMHeaderMeta], str]:
    """Read DICOM header without decoding pixel array.
    
    Returns:
        tuple of (metadata_or_none, reason_code)
    """
    p = str(file_path)
    try:
        ds = pydicom.dcmread(p, stop_before_pixels=True, specific_tags=FAST_HEADER_TAGS)

        series_uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
        desc = str(getattr(ds, "SeriesDescription", "") or "")
        image_type = [str(x).upper() for x in getattr(ds, "ImageType", [])]

        # 1. Hard exclusions
        if any("LOCALIZER" in x for x in image_type):
            return None, "excluded_localizer"
        if _QUANT_MAP_REGEX.search(desc):
            return None, "excluded_regex"

        # 2. Soft heuristics (bypassed if CSV entry exists)
        has_csv_entry = series_metadata is not None and series_uid in series_metadata
        if not has_csv_entry:
            if any("SECONDARY" in x for x in image_type) and not any("PRIMARY" in x for x in image_type):
                return None, "excluded_secondary"
            if _SOFT_EXCLUDE_REGEX.search(desc):
                return None, "excluded_regex"

        inst_num = int(getattr(ds, "InstanceNumber", 0) or 0)
        echo_raw = getattr(ds, "EchoNumbers", getattr(ds, "EchoNumber", 1))
        try:
            echo_num = int(echo_raw) if echo_raw is not None and str(echo_raw).strip() != "" else 1
        except Exception:
            echo_num = 1

        iop_raw = getattr(ds, "ImageOrientationPatient", None)
        ipp_raw = getattr(ds, "ImagePositionPatient", None)

        if iop_raw is not None and ipp_raw is not None and len(iop_raw) == 6 and len(ipp_raw) == 3:
            iop = [float(x) for x in iop_raw]
            ipp = [float(x) for x in ipp_raw]
            k_pos, plane = compute_slice_normal_projection(iop, ipp)
            has_physical = True
        else:
            k_pos = float(inst_num)
            has_physical = False
            plane = "unknown"
            for pl_name, pl_regex in _PLANE_TEXT_REGEX.items():
                if pl_regex.search(desc):
                    plane = pl_name
                    break

        if has_csv_entry:
            _, _, csv_pl = series_metadata[series_uid]
            if csv_pl != "unknown":
                plane = csv_pl

        if plane == "unknown":
            return None, "unknown_plane"

        te = getattr(ds, "EchoTime", None)
        te_val = float(te) if te is not None and te != "" else None
        is_t1 = bool(_T1_REGEX.search(desc))
        is_pd = bool(_PD_REGEX.search(desc)) and not is_t1
        is_non_fs = bool(_NON_FS_PREFIX_REGEX.search(desc))
        is_fs = bool(_FS_REGEX.search(desc)) and not is_non_fs

        is_fluid = (
            (te_val is not None and te_val > 30.0)
            or bool(_FLUID_REGEX.search(desc))
            or (is_fs and not is_t1)
            or is_pd
        )

        if has_csv_entry:
            csv_fl, csv_fs, _ = series_metadata[series_uid]
            if csv_fl is not None:
                is_fluid = bool(csv_fl)
            if csv_fs is not None:
                is_fs = bool(csv_fs)

        ps_raw = getattr(ds, "PixelSpacing", None)
        if ps_raw is not None and len(ps_raw) == 2:
            psy, psx = float(ps_raw[0]), float(ps_raw[1])
            pixel_spacing = (
                psy if psy > 1e-3 else DEFAULT_PIXEL_SPACING[0],
                psx if psx > 1e-3 else DEFAULT_PIXEL_SPACING[1],
            )
        else:
            pixel_spacing = DEFAULT_PIXEL_SPACING

        meta = DICOMHeaderMeta(
            path=p,
            series_uid=series_uid,
            k_pos=k_pos,
            has_physical_coords=has_physical,
            plane=plane,
            is_fluid=is_fluid,
            is_fs=is_fs,
            pixel_spacing=pixel_spacing,
            instance_number=inst_num,
            echo_number=echo_num,
        )
        return meta, "ok"
    except Exception:
        return None, "parse_error"


def route_series_to_slot(plane: str, is_fluid: bool, is_fs: bool) -> Optional[int]:
    """Map series metadata to slot index 0..5."""
    plane = normalize_plane_str(plane)
    if plane == "sagittal":
        if is_fluid and is_fs:
            return 0  # SAG_FLUID_FS
        elif is_fluid and not is_fs:
            return 3  # SAG_FLUID_NOFS
        else:
            return 5  # SAG_T1
    elif plane == "coronal":
        if is_fluid:
            return 1  # COR_FLUID_FS
        else:
            return 4  # COR_T1
    elif plane == "axial":
        if is_fluid:
            return 2  # AX_FLUID_FS
    return None


def crop_and_resize_140mm(
    img_triplet: np.ndarray,
    pixel_spacing: tuple[float, float],
    target_size: int = 336,
    crop_mm: float = DEFAULT_CROP_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract square 140 mm physical FOV centered on image array and resize to target_size x target_size.
    
    Returns:
        tuple of (cropped_triplet, unpadded_valid_mask)
        unpadded_valid_mask is strictly resized to (target_size, target_size) boolean.
    """
    _, h, w = img_triplet.shape
    psy = max(float(pixel_spacing[0]), 1e-3)
    psx = max(float(pixel_spacing[1]), 1e-3)

    req_h = int(round(crop_mm / psy))
    req_w = int(round(crop_mm / psx))

    pad_y = max(0, req_h - h)
    pad_x = max(0, req_w - w)

    valid_mask = np.ones((h, w), dtype=bool)

    if pad_y > 0 or pad_x > 0:
        pad_top = pad_y // 2
        pad_bot = pad_y - pad_top
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left
        img_triplet = np.pad(
            img_triplet,
            ((0, 0), (pad_top, pad_bot), (pad_left, pad_right)),
            mode="constant",
            constant_values=0.0,
        )
        valid_mask = np.pad(
            valid_mask,
            ((pad_top, pad_bot), (pad_left, pad_right)),
            mode="constant",
            constant_values=False,
        )
        _, h, w = img_triplet.shape

    y0 = max(0, (h - req_h) // 2)
    x0 = max(0, (w - req_w) // 2)
    cropped = img_triplet[:, y0 : y0 + req_h, x0 : x0 + req_w]
    cropped_mask = valid_mask[y0 : y0 + req_h, x0 : x0 + req_w]

    out = np.empty((3, target_size, target_size), dtype=np.float32)
    interp = cv2.INTER_AREA if (req_h >= target_size and req_w >= target_size) else cv2.INTER_LINEAR
    for i in range(3):
        out[i] = cv2.resize(cropped[i], (target_size, target_size), interpolation=interp)

    out_mask = cv2.resize(
        cropped_mask.astype(np.uint8),
        (target_size, target_size),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    return out, out_mask


def normalize_triplet_dinov2(
    triplet: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, bool]:
    """Per-volume percentile normalization [0.5, 99.5] followed by ImageNet standardization."""
    if valid_mask is not None and np.any(valid_mask):
        unpadded_pixels = triplet[:, valid_mask]
    else:
        unpadded_pixels = triplet

    p_low, p_high = np.percentile(unpadded_pixels, [0.5, 99.5])

    if p_high <= p_low:
        return np.zeros_like(triplet, dtype=np.float32), False

    normed = np.clip((triplet - p_low) / (p_high - p_low), 0.0, 1.0)
    standardized = (normed - IMAGENET_MEAN) / IMAGENET_STD
    cleaned = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return cleaned, True


def deduplicate_multi_echo_slices(headers: Sequence[DICOMHeaderMeta]) -> list[DICOMHeaderMeta]:
    """Collapse multi-echo slices sharing the same physical position (|k1 - k2| < 0.25mm) to lowest echo in O(N log N)."""
    if not headers:
        return []
    sorted_h = sorted(headers, key=lambda x: (x.k_pos, x.echo_number, x.instance_number))
    clusters: list[DICOMHeaderMeta] = [sorted_h[0]]
    for h in sorted_h[1:]:
        if abs(h.k_pos - clusters[-1].k_pos) < 0.25:
            if h.echo_number < clusters[-1].echo_number:
                clusters[-1] = h
        else:
            clusters.append(h)
    return clusters


def select_triplet_paths(
    headers: Sequence[DICOMHeaderMeta],
    slice_pos: float = 0.5,
) -> list[str]:
    """Select adjacent 2.5D triplet [c-1, c, c+1] after sorting slices by canonical physical k."""
    deduped = deduplicate_multi_echo_slices(headers)
    sorted_h = sorted(deduped, key=lambda x: (x.k_pos, x.instance_number, x.path))
    n = len(sorted_h)
    if n == 0:
        return []
    if n == 1:
        return [sorted_h[0].path]
    if n == 2:
        return [sorted_h[0].path, sorted_h[0].path, sorted_h[1].path]

    c_idx = max(0, min(n - 1, int(round((n - 1) * slice_pos))))
    c_k = sorted_h[c_idx].k_pos

    if sorted_h[0].has_physical_coords and n >= 5:
        k_diffs = [
            abs(sorted_h[i + 1].k_pos - sorted_h[i].k_pos)
            for i in range(n - 1)
            if abs(sorted_h[i + 1].k_pos - sorted_h[i].k_pos) > 1e-4
        ]
        med_spacing = float(np.median(k_diffs)) if k_diffs else 3.0
        if med_spacing < 1.5:
            left_idx = min(range(n), key=lambda i: abs((sorted_h[i].k_pos - c_k) - (-3.0)))
            right_idx = min(range(n), key=lambda i: abs((sorted_h[i].k_pos - c_k) - 3.0))
            if left_idx == c_idx:
                left_idx = max(0, c_idx - 1)
            if right_idx == c_idx:
                right_idx = min(n - 1, c_idx + 1)
            return [sorted_h[left_idx].path, sorted_h[c_idx].path, sorted_h[right_idx].path]

    return [sorted_h[max(0, c_idx - 1)].path, sorted_h[c_idx].path, sorted_h[min(n - 1, c_idx + 1)].path]


def decode_triplet_slices(paths: list[str]) -> Optional[np.ndarray]:
    """Decode pixel array strictly for the target slices, avoiding duplicate I/O on replicated slices."""
    if not paths:
        return None

    unique_paths = list(dict.fromkeys(paths))
    loaded: dict[str, np.ndarray] = {}
    base_shape = None

    for p in unique_paths:
        ds = pydicom.dcmread(p)
        arr = ds.pixel_array.astype(np.float32)
        if base_shape is None:
            base_shape = arr.shape
        elif arr.shape != base_shape:
            return None

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        loaded[p] = arr * slope + intercept

    if len(paths) == 1:
        single_arr = loaded[paths[0]]
        return np.repeat(single_arr[np.newaxis, ...], 3, axis=0)

    arrays = [loaded[p] for p in paths]
    return np.stack(arrays, axis=0)


def process_study_to_6slot(
    study_dir: str | Path,
    series_metadata: Optional[dict[str, tuple[Optional[int], Optional[int], str]]] = None,
    num_workers: int = 4,
    target_size: int = 336,
    crop_mm: float = DEFAULT_CROP_MM,
    slice_pos: float = 0.5,
    return_stats: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Process all series in a study directory into 6-slot tensor and presence mask."""
    study_path = Path(study_dir)
    dcm_files = [str(f) for f in sorted(study_path.rglob("*")) if f.is_file() and not f.name.startswith(".")]

    study_tensor = np.zeros((NUM_SLOTS, 3, target_size, target_size), dtype=np.float32)
    presence_mask = np.zeros(NUM_SLOTS, dtype=np.float32)
    stats: dict[str, Any] = {
        "raw_files": len(dcm_files),
        "parsed_ok": 0,
        "excluded_localizer": 0,
        "excluded_secondary": 0,
        "excluded_regex": 0,
        "unknown_plane": 0,
        "unrouted_series": 0,
        "parse_error": 0,
        "short_series": 0,
        "decode_errors": 0,
        "constant_volumes": 0,
        "decode_error_types": {},
        "selected_series_uids": {},
    }

    if not dcm_files:
        if return_stats:
            return torch.from_numpy(study_tensor), torch.from_numpy(presence_mask), stats
        return torch.from_numpy(study_tensor), torch.from_numpy(presence_mask)

    headers: list[DICOMHeaderMeta] = []
    if num_workers > 0:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            fn = lambda f: parse_dicom_header_fast(f, series_metadata=series_metadata)
            for meta, code in executor.map(fn, dcm_files):
                if meta is not None:
                    headers.append(meta)
                    stats["parsed_ok"] += 1
                else:
                    stats[code] = stats.get(code, 0) + 1
    else:
        for f in dcm_files:
            meta, code = parse_dicom_header_fast(f, series_metadata=series_metadata)
            if meta is not None:
                headers.append(meta)
                stats["parsed_ok"] += 1
            else:
                stats[code] = stats.get(code, 0) + 1

    if not headers:
        if return_stats:
            return torch.from_numpy(study_tensor), torch.from_numpy(presence_mask), stats
        return torch.from_numpy(study_tensor), torch.from_numpy(presence_mask)

    series_groups: dict[str, list[DICOMHeaderMeta]] = {}
    for h in headers:
        series_groups.setdefault(h.series_uid, []).append(h)

    slot_cands: dict[int, list[tuple[tuple[int, int, int], str, list[DICOMHeaderMeta]]]] = {}

    for s_uid in sorted(series_groups.keys()):
        s_headers = series_groups[s_uid]
        sample = s_headers[0]

        slot = route_series_to_slot(sample.plane, sample.is_fluid, sample.is_fs)
        is_fs_flag = int(sample.is_fs)

        if slot is not None:
            deduped_slices = deduplicate_multi_echo_slices(s_headers)
            unique_k_cnt = len(deduped_slices)
            if unique_k_cnt < 3:
                stats["short_series"] += 1
                continue

            k_vals = sorted([h.k_pos for h in deduped_slices])
            diffs = [abs(k_vals[i + 1] - k_vals[i]) for i in range(len(k_vals) - 1) if abs(k_vals[i + 1] - k_vals[i]) > 1e-4]
            med_sp = float(np.median(diffs)) if diffs else 3.0

            if sample.has_physical_coords:
                is_2d = 1 if (1.5 <= med_sp <= 6.0 and 10 <= unique_k_cnt <= 80) else 0
            else:
                is_2d = 1 if (10 <= unique_k_cnt <= 80) else 0

            fs_pri = is_fs_flag if slot in (0, 1, 2) else 0

            score = (-fs_pri, -is_2d, -unique_k_cnt)
            slot_cands.setdefault(slot, []).append((score, s_uid, s_headers))
        else:
            stats["unrouted_series"] += len(s_headers)

    for slot_idx, cands in slot_cands.items():
        cands.sort(key=lambda c: (c[0], c[1]))
        for _, uid, s_headers in cands:
            triplet_paths = select_triplet_paths(s_headers, slice_pos=slice_pos)
            if not triplet_paths:
                continue
            try:
                raw_triplet = decode_triplet_slices(triplet_paths)
                if raw_triplet is None:
                    stats["decode_errors"] += 1
                    continue

                cropped_triplet, valid_mask = crop_and_resize_140mm(
                    raw_triplet,
                    pixel_spacing=s_headers[0].pixel_spacing,
                    target_size=target_size,
                    crop_mm=crop_mm,
                )
                normed_triplet, is_valid = normalize_triplet_dinov2(cropped_triplet, valid_mask=valid_mask)
                if not is_valid:
                    stats["constant_volumes"] += 1
                    continue

                study_tensor[slot_idx] = normed_triplet
                presence_mask[slot_idx] = 1.0
                stats["selected_series_uids"][slot_idx] = uid
                break
            except Exception as e:
                stats["decode_errors"] += 1
                err_type = type(e).__name__
                stats["decode_error_types"][err_type] = stats["decode_error_types"].get(err_type, 0) + 1

    if return_stats:
        return torch.from_numpy(study_tensor), torch.from_numpy(presence_mask), stats
    return torch.from_numpy(study_tensor), torch.from_numpy(presence_mask)


class Fast6SlotDICOMDataset(Dataset):
    """PyTorch Dataset loading 6-slot studies via fast 18-slice selective decoding."""

    def __init__(
        self,
        study_dirs: Sequence[str | Path],
        labels: Optional[np.ndarray | Sequence[Sequence[float]]] = None,
        series_csv_path: Optional[str | Path] = None,
        num_workers: int = 0,
        target_size: int = 336,
        crop_mm: float = DEFAULT_CROP_MM,
        slice_pos: float = 0.5,
    ) -> None:
        self.study_dirs = [Path(d) for d in study_dirs]
        self.labels = np.array(labels, dtype=np.float32) if labels is not None else None
        self.num_workers = num_workers
        self.target_size = target_size
        self.crop_mm = crop_mm
        self.slice_pos = slice_pos
        self.series_metadata = load_series_metadata(series_csv_path) if series_csv_path else None

    def __len__(self) -> int:
        return len(self.study_dirs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        study_dir = self.study_dirs[idx]
        tensor, mask = process_study_to_6slot(
            study_dir,
            series_metadata=self.series_metadata,
            num_workers=self.num_workers,
            target_size=self.target_size,
            crop_mm=self.crop_mm,
            slice_pos=self.slice_pos,
        )
        item: dict[str, Any] = {
            "study_id": study_dir.name,
            "image": tensor,
            "presence_mask": mask,
        }
        if self.labels is not None:
            item["labels"] = torch.from_numpy(self.labels[idx])
        return item
