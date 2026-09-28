"""
tests/test_efficiency_geometry.py

Unit tests verifying DICOM sign-canonical normal projection sorting,
physical 140 mm FOV scale invariance (including <140 mm FOV padding),
constant volume safety, collision ranking with FS and 2D diagnostic priority,
runner-up fallback, non-FS PD/PDW header routing, CSV NaN fallback,
and Fast6SlotDICOMDataset contracts.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pydicom
import pytest
import torch
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from src.datasets.efficiency_pipeline import (
    NUM_SLOTS,
    Fast6SlotDICOMDataset,
    compute_slice_normal_projection,
    crop_and_resize_140mm,
    decode_triplet_slices,
    load_series_metadata,
    normalize_triplet_dinov2,
    parse_dicom_header_fast,
    process_study_to_6slot,
    route_series_to_slot,
    select_triplet_paths,
    DICOMHeaderMeta,
)


def create_synthetic_dicom_file(
    file_path: Path,
    study_uid: str,
    series_uid: str,
    instance_number: int,
    iop: list[float],
    ipp: list[float],
    pixel_spacing: list[float],
    series_desc: str = "SAG PD FS",
    echo_time: float = 35.0,
    echo_numbers: int = 1,
    image_type: list[str] | None = None,
    pixel_array: np.ndarray | None = None,
) -> Path:
    """Create a minimal synthetic valid DICOM file on disk with non-constant pixels."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(file_path), {}, file_meta=file_meta, preamble=b"\0" * 128)

    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.InstanceNumber = instance_number
    ds.SeriesDescription = series_desc
    ds.EchoTime = echo_time
    ds.EchoNumbers = echo_numbers
    ds.ImageOrientationPatient = iop
    ds.ImagePositionPatient = ipp
    ds.PixelSpacing = pixel_spacing
    ds.ImageType = image_type if image_type is not None else ["ORIGINAL", "PRIMARY", "M_SE"]
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0

    if pixel_array is None:
        y, x = np.mgrid[0:100, 0:100]
        pixel_array = (y * 2 + x * 3).astype(np.uint16)
    else:
        pixel_array = pixel_array.astype(np.uint16)

    ds.Rows, ds.Columns = pixel_array.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = pixel_array.tobytes()

    ds.save_as(str(file_path))
    return file_path


def test_normal_projection_monotonicity_under_reversed_instance_number():
    """Assert k-projection sorts slices strictly monotonically even if InstanceNumbers are inverted."""
    iop = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    z_coords = [0.0, 3.0, 6.0, 9.0, 12.0]
    inverted_instances = [50, 40, 30, 20, 10]

    headers: list[DICOMHeaderMeta] = []
    for z, inst in zip(z_coords, inverted_instances):
        ipp = [0.0, 0.0, z]
        k, plane = compute_slice_normal_projection(iop, ipp)
        headers.append(
            DICOMHeaderMeta(
                path=f"slice_z_{z}.dcm",
                series_uid="series_1",
                k_pos=k,
                has_physical_coords=True,
                plane=plane,
                is_fluid=True,
                is_fs=True,
                pixel_spacing=(0.4, 0.4),
                instance_number=inst,
            )
        )

    np.random.seed(42)
    np.random.shuffle(headers)

    selected = select_triplet_paths(headers)
    assert selected == ["slice_z_3.0.dcm", "slice_z_6.0.dcm", "slice_z_9.0.dcm"]


def test_multi_echo_deduplication_and_unique_k():
    """Assert multi-echo duplicate positions (|k1 - k2| < 0.25mm) collapse to lowest echo."""
    headers: list[DICOMHeaderMeta] = []
    positions = [10.0, 20.0, 30.0]
    for p in positions:
        headers.append(
            DICOMHeaderMeta(
                path=f"pos_{p}_e1.dcm",
                series_uid="me_series",
                k_pos=p,
                has_physical_coords=True,
                plane="sagittal",
                is_fluid=True,
                is_fs=True,
                pixel_spacing=(0.4, 0.4),
                instance_number=int(p),
                echo_number=1,
            )
        )
        headers.append(
            DICOMHeaderMeta(
                path=f"pos_{p}_e2.dcm",
                series_uid="me_series",
                k_pos=p + 0.05,
                has_physical_coords=True,
                plane="sagittal",
                is_fluid=True,
                is_fs=True,
                pixel_spacing=(0.4, 0.4),
                instance_number=int(p) + 50,
                echo_number=2,
            )
        )

    triplet = select_triplet_paths(headers)
    assert len(triplet) == 3
    assert triplet == ["pos_10.0_e1.dcm", "pos_20.0_e1.dcm", "pos_30.0_e1.dcm"]


def test_synthetic_header_pd_routing():
    """Assert synthetic DICOM with SAG PD FSE and TE=27ms parses as fluid and non-FS (Slot 3)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dcm_file = Path(tmp_dir) / "test_pd.dcm"
        create_synthetic_dicom_file(
            dcm_file,
            study_uid=generate_uid(),
            series_uid=generate_uid(),
            instance_number=1,
            iop=[0.0, 1.0, 0.0, 0.0, 0.0, -1.0],  # Sagittal
            ipp=[10.0, 0.0, 0.0],
            pixel_spacing=[0.4, 0.4],
            series_desc="SAG PD FSE",
            echo_time=27.0,
        )
        meta, code = parse_dicom_header_fast(dcm_file)
        assert code == "ok"
        assert meta is not None
        assert meta.plane == "sagittal"
        assert meta.is_fluid is True
        assert meta.is_fs is False
        slot = route_series_to_slot(meta.plane, meta.is_fluid, meta.is_fs)
        assert slot == 3  # SAG_FLUID_NOFS


def test_pdw_and_t2w_regex_routing():
    """Assert PDW TSE and T2W TSE correctly match and route without word-boundary failure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        p1 = Path(tmp_dir) / "pdw.dcm"
        create_synthetic_dicom_file(
            p1,
            study_uid=generate_uid(),
            series_uid=generate_uid(),
            instance_number=1,
            iop=[0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            ipp=[0.0, 0.0, 0.0],
            pixel_spacing=[0.4, 0.4],
            series_desc="SAG PDW TSE",
            echo_time=25.0,
        )
        m1, c1 = parse_dicom_header_fast(p1)
        assert c1 == "ok"
        assert m1 is not None and m1.is_fluid is True and m1.is_fs is False
        assert route_series_to_slot(m1.plane, m1.is_fluid, m1.is_fs) == 3

        p2 = Path(tmp_dir) / "t2w.dcm"
        create_synthetic_dicom_file(
            p2,
            study_uid=generate_uid(),
            series_uid=generate_uid(),
            instance_number=1,
            iop=[1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            ipp=[0.0, 0.0, 0.0],
            pixel_spacing=[0.4, 0.4],
            series_desc="COR T2W TSE",
            echo_time=50.0,
        )
        m2, c2 = parse_dicom_header_fast(p2)
        assert c2 == "ok"
        assert m2 is not None and m2.is_fluid is True
        assert route_series_to_slot(m2.plane, m2.is_fluid, m2.is_fs) == 1


def test_collision_ranking_fs_and_2d_diagnostic_priority():
    """Assert FS priority and 2D diagnostic preference resolve slot collisions correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        study_dir = Path(tmp_dir) / "study_collision"
        study_dir.mkdir()

        sA = study_dir / "series_non_fs"
        sA.mkdir()
        sA_uid = generate_uid()
        for i in range(50):
            create_synthetic_dicom_file(
                sA / f"s_{i}.dcm",
                study_uid=generate_uid(),
                series_uid=sA_uid,
                instance_number=i + 1,
                iop=[1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                ipp=[0.0, float(i * 3), 0.0],
                pixel_spacing=[0.4, 0.4],
                series_desc="COR T2 NON-FS",
                echo_time=45.0,
            )

        sB = study_dir / "series_fs"
        sB.mkdir()
        sB_uid = generate_uid()
        for i in range(30):
            create_synthetic_dicom_file(
                sB / f"s_{i}.dcm",
                study_uid=generate_uid(),
                series_uid=sB_uid,
                instance_number=i + 1,
                iop=[1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                ipp=[0.0, float(i * 3), 0.0],
                pixel_spacing=[0.4, 0.4],
                series_desc="COR T2 FAT SAT",
                echo_time=45.0,
            )

        tensor, mask, stats = process_study_to_6slot(study_dir, num_workers=0, return_stats=True)
        assert mask[1].item() == 1.0
        assert stats["selected_series_uids"][1] == sB_uid


def test_runner_up_used_when_winner_is_constant():
    """Assert runner-up candidate series fills the slot when the top-ranked candidate is blank/constant."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        study_dir = Path(tmp_dir) / "study_runner_up"
        study_dir.mkdir()

        # Series A: top-ranked (FS, 30 slices), but ALL CONSTANT PIXELS
        sA = study_dir / "series_const"
        sA.mkdir()
        sA_uid = generate_uid()
        const_arr = np.full((100, 100), 100, dtype=np.uint16)
        for i in range(30):
            create_synthetic_dicom_file(
                sA / f"s_{i}.dcm",
                study_uid=generate_uid(),
                series_uid=sA_uid,
                instance_number=i + 1,
                iop=[1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                ipp=[0.0, float(i * 3), 0.0],
                pixel_spacing=[0.4, 0.4],
                series_desc="COR T2 FS CONSTANT",
                echo_time=45.0,
                pixel_array=const_arr,
            )

        # Series B: lower-ranked (non-FS, 20 slices), but VALID NON-CONSTANT PIXELS
        sB = study_dir / "series_valid"
        sB.mkdir()
        sB_uid = generate_uid()
        for i in range(20):
            create_synthetic_dicom_file(
                sB / f"s_{i}.dcm",
                study_uid=generate_uid(),
                series_uid=sB_uid,
                instance_number=i + 1,
                iop=[1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                ipp=[0.0, float(i * 3), 0.0],
                pixel_spacing=[0.4, 0.4],
                series_desc="COR T2 VALID",
                echo_time=45.0,
            )

        tensor, mask, stats = process_study_to_6slot(study_dir, num_workers=0, return_stats=True)
        assert mask[1].item() == 1.0
        assert stats["selected_series_uids"][1] == sB_uid
        assert stats["constant_volumes"] >= 1


def test_140mm_physical_fov_scale_invariance_with_half_max():
    """Assert 140 mm crop produces identical pixel feature scale regardless of PixelSpacing (0.35 vs 0.70 mm)."""
    target_size = 336

    ps_a = (0.35, 0.35)
    img_a = np.zeros((3, 400, 400), dtype=np.float32)
    img_a[:, 100:300, 100:300] = 1000.0

    ps_b = (0.70, 0.70)
    img_b = np.zeros((3, 200, 200), dtype=np.float32)
    img_b[:, 50:150, 50:150] = 1000.0

    crop_a, mask_a = crop_and_resize_140mm(img_a, ps_a, target_size=target_size, crop_mm=140.0)
    crop_b, mask_b = crop_and_resize_140mm(img_b, ps_b, target_size=target_size, crop_mm=140.0)

    assert mask_a.shape == (target_size, target_size)
    assert mask_b.shape == (target_size, target_size)

    active_a = np.where(crop_a[0] >= 500.0)
    active_b = np.where(crop_b[0] >= 500.0)

    span_a = (active_a[0].max() - active_a[0].min() + 1, active_a[1].max() - active_a[1].min() + 1)
    span_b = (active_b[0].max() - active_b[0].min() + 1, active_b[1].max() - active_b[1].min() + 1)

    assert abs(span_a[0] - span_b[0]) <= 2
    assert abs(span_a[0] - 168) <= 2


def test_small_fov_under_140mm_zero_padding_and_mask_shape():
    """Assert images smaller than 140 mm physical FOV are zero-padded and mask matches target size."""
    target_size = 336
    ps = (0.5, 0.5)
    img_small = np.zeros((3, 200, 200), dtype=np.float32)
    img_small[:, 50:150, 50:150] = 1000.0

    cropped, mask = crop_and_resize_140mm(img_small, ps, target_size=target_size, crop_mm=140.0)
    assert cropped.shape == (3, target_size, target_size)
    assert mask.shape == (target_size, target_size)
    assert mask.dtype == bool

    normed, is_valid = normalize_triplet_dinov2(cropped, valid_mask=mask)
    assert is_valid is True
    assert normed.shape == (3, target_size, target_size)
    assert np.all(np.isfinite(normed))


def test_3d_isotropic_sampling_step():
    """Assert thin 3D isotropic acquisition (spacing 0.5mm, N>=5) samples ~3.0mm apart."""
    headers: list[DICOMHeaderMeta] = []
    for i in range(21):
        z = float(i * 0.5)
        headers.append(
            DICOMHeaderMeta(
                path=f"slice_{z:.1f}.dcm",
                series_uid="iso_3d",
                k_pos=z,
                has_physical_coords=True,
                plane="sagittal",
                is_fluid=True,
                is_fs=True,
                pixel_spacing=(0.5, 0.5),
                instance_number=i + 1,
            )
        )
    triplet = select_triplet_paths(headers, slice_pos=0.5)
    assert triplet == ["slice_2.0.dcm", "slice_5.0.dcm", "slice_8.0.dcm"]


def test_csv_nan_falls_back_to_heuristics():
    """Assert NaN in CSV does not force T1 and allows DICOM physics/regex heuristics to operate."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "test_series.csv"
        uid = generate_uid()
        df = pd.DataFrame([{
            "SeriesInstanceUID": uid,
            "Fluid_Sensitive": np.nan,
            "Fat_Suppression": 1,
            "Anatomical_Plane": "Sagittal",
        }])
        df.to_csv(csv_path, index=False)
        meta_dict = load_series_metadata(csv_path)
        assert meta_dict is not None
        assert meta_dict[uid][0] is None

        dcm_p = Path(tmp_dir) / "test.dcm"
        create_synthetic_dicom_file(
            dcm_p,
            study_uid=generate_uid(),
            series_uid=uid,
            instance_number=1,
            iop=[0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            ipp=[0.0, 0.0, 0.0],
            pixel_spacing=[0.4, 0.4],
            series_desc="SAG T2 FS",
            echo_time=45.0,
        )
        meta, code = parse_dicom_header_fast(dcm_p, series_metadata=meta_dict)
        assert code == "ok"
        assert meta is not None
        assert meta.is_fluid is True
        assert meta.is_fs is True


def test_derived_secondary_with_csv_kept():
    """Assert a SECONDARY capture with CSV entry is kept and not dropped by heuristics."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        uid = generate_uid()
        csv_path = Path(tmp_dir) / "series.csv"
        pd.DataFrame([{
            "SeriesInstanceUID": uid,
            "Fluid_Sensitive": 1,
            "Fat_Suppression": 1,
            "Anatomical_Plane": "Coronal",
        }]).to_csv(csv_path, index=False)
        meta_dict = load_series_metadata(csv_path)

        dcm_p = Path(tmp_dir) / "sec.dcm"
        create_synthetic_dicom_file(
            dcm_p,
            study_uid=generate_uid(),
            series_uid=uid,
            instance_number=1,
            iop=[1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            ipp=[0.0, 0.0, 0.0],
            pixel_spacing=[0.4, 0.4],
            series_desc="COR T2 REFORMAT",
            echo_time=40.0,
            image_type=["DERIVED", "SECONDARY"],
        )
        meta, code = parse_dicom_header_fast(dcm_p, series_metadata=meta_dict)
        assert code == "ok"
        assert meta is not None


def test_constant_volume_demotion():
    """Verify constant volume division-by-zero protection returns exact zeros and is_valid=False."""
    const_triplet = np.full((3, 100, 100), 255.0, dtype=np.float32)
    normed, is_valid = normalize_triplet_dinov2(const_triplet)
    assert is_valid is False
    assert np.all(normed == 0.0)
    assert np.all(np.isfinite(normed))


def test_dataset_class_contract():
    """Verify Fast6SlotDICOMDataset __getitem__ returns study_id, image, presence_mask, and labels."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        study_dir = Path(tmp_dir) / "study_test_dataset"
        study_dir.mkdir()
        s_dir = study_dir / "series_0"
        s_dir.mkdir()
        s_uid = generate_uid()

        for i in range(3):
            create_synthetic_dicom_file(
                s_dir / f"slice_{i}.dcm",
                study_uid=generate_uid(),
                series_uid=s_uid,
                instance_number=i + 1,
                iop=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                ipp=[0.0, 0.0, float(i * 3)],
                pixel_spacing=[0.4, 0.4],
                series_desc="AX T2 FS",
                echo_time=40.0,
            )

        labels = np.zeros((1, 12), dtype=np.float32)
        dataset = Fast6SlotDICOMDataset(
            study_dirs=[study_dir],
            labels=labels,
            num_workers=0,
            target_size=336,
        )

        assert len(dataset) == 1
        item = dataset[0]

        assert "study_id" in item and item["study_id"] == "study_test_dataset"
        assert "image" in item and item["image"].shape == (6, 3, 336, 336)
        assert "presence_mask" in item and item["presence_mask"].shape == (6,)
        assert "labels" in item and item["labels"].shape == (12,)


def test_full_study_6slot_processing_and_absent_slot_mask():
    """Verify end-to-end 6-slot tensor construction with absent slot zero-gating."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        study_dir = Path(tmp_dir) / "study_001"
        study_dir.mkdir()

        s1_dir = study_dir / "series_sag_fs"
        s1_dir.mkdir()
        s1_uid = generate_uid()
        for i in range(3):
            create_synthetic_dicom_file(
                s1_dir / f"slice_{i}.dcm",
                study_uid=generate_uid(),
                series_uid=s1_uid,
                instance_number=i + 1,
                iop=[0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
                ipp=[float(i * 3), 0.0, 0.0],
                pixel_spacing=[0.4, 0.4],
                series_desc="SAG PDFS",
                echo_time=40.0,
            )

        s2_dir = study_dir / "series_cor_t1"
        s2_dir.mkdir()
        s2_uid = generate_uid()
        for i in range(3):
            create_synthetic_dicom_file(
                s2_dir / f"slice_{i}.dcm",
                study_uid=generate_uid(),
                series_uid=s2_uid,
                instance_number=i + 1,
                iop=[1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                ipp=[0.0, float(i * 3), 0.0],
                pixel_spacing=[0.4, 0.4],
                series_desc="COR T1",
                echo_time=15.0,
            )

        tensor, mask, stats = process_study_to_6slot(study_dir, num_workers=0, return_stats=True)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (6, 3, 336, 336)
        assert tensor.dtype == torch.float32

        assert isinstance(mask, torch.Tensor)
        assert mask.shape == (6,)
        assert mask.dtype == torch.float32

        expected_mask = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
        assert torch.equal(mask, expected_mask)

        # Absent slots strictly zeros
        assert torch.all(tensor[1] == 0.0)
        assert torch.all(tensor[2] == 0.0)
        assert torch.all(tensor[3] == 0.0)
        assert torch.all(tensor[5] == 0.0)

        # Present slots contain normalized values
        assert not torch.all(tensor[0] == 0.0)
        assert not torch.all(tensor[4] == 0.0)
        assert torch.all(torch.isfinite(tensor))
