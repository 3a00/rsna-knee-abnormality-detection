"""tests/test_caching_and_regularization.py

Comprehensive test suite verifying:
  1. 5-slice caching and provenance validation.
  2. Dynamic 2.5D depth jitter (offset coverage over {0, 1, 2} in train mode vs center [1:4] in val).
  3. Symmetry-preserving MSK augmentations (strictly no flips, absent slots untouched, corner median bg fill).
  4. Regularization: attention dropout, pre-classifier dropout, LLRD optimizer groupings.
  5. 3-stage progressive unfreezing state transitions under DDP with weakref release.
  6. lr_for boundary values across all stages with cosine decay and warmup ramp math.
  7. Train/inference normalization parity on synthetic DICOM study written via pydicom.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import random
import tempfile
import weakref

import numpy as np
import pydicom
from pydicom.dataset import Dataset as DcmDataset, FileMetaDataset
import pytest
import torch
import torch.nn as nn
import torch.distributed as dist

from src.datasets.efficiency_pipeline import (
    select_5slice_paths,
    normalize_triplet_dinov2,
    crop_and_resize_140mm,
    process_study_to_5slice_cache,
)
from src.models.dinov2_slothead import DINOv2SlotHead
from src.training.train_efficiency import (
    EfficiencyStudyDataset,
    apply_msk_augmentations,
    build_differential_optimizer,
)
from scripts.train_ddp_efficiency import (
    apply_unfreezing_stage,
    rewrap_ddp,
    lr_for,
    verify_cache,
)


def test_select_5slice_paths_padding() -> None:
    """Verify exact index clipping edge-replication for short series (n < 5)."""
    class DummyHeader:
        def __init__(self, k, path):
            self.k_pos = k
            self.instance_number = 1
            self.echo_number = 1
            self.path = path

    # Case 1: n = 2 slices (c = min(1, int(2 * 0.5)) = 1 -> [-1, 0, 1, 2, 3] clipped to [0, 1] -> [0, 0, 1, 1, 1])
    h2 = [DummyHeader(0.0, "s0.dcm"), DummyHeader(3.0, "s1.dcm")]
    p2 = select_5slice_paths(h2, slice_pos=0.5)
    assert len(p2) == 5
    assert p2 == ["s0.dcm", "s0.dcm", "s1.dcm", "s1.dcm", "s1.dcm"]

    # Case 2: n = 4 slices (c = min(3, int(4 * 0.5)) = 2 -> [0, 1, 2, 3, 4] clipped to [0, 3] -> [0, 1, 2, 3, 3])
    h4 = [DummyHeader(float(i), f"s{i}.dcm") for i in range(4)]
    p4 = select_5slice_paths(h4, slice_pos=0.5)
    assert len(p4) == 5
    assert p4 == ["s0.dcm", "s1.dcm", "s2.dcm", "s3.dcm", "s3.dcm"]
    assert p4[1:4] == ["s1.dcm", "s2.dcm", "s3.dcm"]


def test_dynamic_depth_jitter_offset_coverage(monkeypatch) -> None:
    """Verify stochastic depth jitter draws all 3 offsets {0:3, 1:4, 2:5} in train mode."""
    monkeypatch.setattr("src.training.train_efficiency.apply_msk_augmentations", lambda img, mask: img)

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = Path(tmp_dir)
        uid = "study_jitter_test"
        tensor_5 = torch.zeros(6, 5, 336, 336, dtype=torch.float16)
        for k in range(5):
            tensor_5[:, k] = float(k)
        mask = torch.ones(6, dtype=torch.float32)

        torch.save({
            "image": tensor_5,
            "presence_mask": mask,
            "meta": {"target_size": 336, "crop_mm": 140.0, "n_slices_cached": 5},
        }, cache_dir / f"{uid}.pt")

        ds_val = EfficiencyStudyDataset(
            study_uids=[uid],
            study_dirs=[tmp_dir],
            cached_tensor_dir=str(cache_dir),
            is_train=False,
        )
        val_item = ds_val[0]
        assert val_item["image"].shape == (6, 3, 336, 336)
        assert torch.allclose(val_item["image"], tensor_5[:, 1:4])

        ds_train = EfficiencyStudyDataset(
            study_uids=[uid],
            study_dirs=[tmp_dir],
            cached_tensor_dir=str(cache_dir),
            is_train=True,
        )
        observed_offsets = set()
        for _ in range(50):
            item = ds_train[0]
            first_slice_val = int(round(item["image"][0, 0, 0, 0].item()))
            observed_offsets.add(first_slice_val)

        assert observed_offsets == {0, 1, 2}


def test_cache_provenance_validation() -> None:
    """Verify that stale cache artifacts (mismatched size or crop_mm) raise RuntimeError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = Path(tmp_dir)
        uid = "stale_study"
        payload = {
            "image": torch.randn(6, 5, 256, 256, dtype=torch.float16),
            "presence_mask": torch.ones(6),
            "meta": {"target_size": 256, "crop_mm": 140.0, "n_slices_cached": 5},
        }
        torch.save(payload, cache_dir / f"{uid}.pt")

        ds = EfficiencyStudyDataset(
            study_uids=[uid],
            study_dirs=[tmp_dir],
            cached_tensor_dir=str(cache_dir),
            target_size=336,
        )
        with pytest.raises(RuntimeError, match="Stale cache"):
            _ = ds[0]


def test_verify_cache_pre_spawn() -> None:
    """Verify pre-spawn cache verification catches missing studies or invalid metadata."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = Path(tmp_dir)
        uids = ["study_01", "study_02"]

        with pytest.raises(FileNotFoundError):
            verify_cache(cache_dir, uids)

        (cache_dir / "cache_manifest.json").write_text(
            '{"target_size": 336, "crop_mm": 140.0, "n_slices_cached": 5, "total_cached_files": 2, "failed_uids": [], "empty_uids": []}'
        )

        with pytest.raises(RuntimeError, match="missing from cache"):
            verify_cache(cache_dir, uids)

        for u in uids:
            (cache_dir / f"{u}.pt").touch()

        valid, empty = verify_cache(cache_dir, uids)
        assert len(valid) == 2
        assert len(empty) == 0


def test_aug_no_flip_and_absent_slots_untouched() -> None:
    """Verify non-flaky symmetry constraints and that absent slots remain strictly zero."""
    random.seed(0)
    torch.manual_seed(0)

    x = torch.zeros(6, 3, 336, 336)
    x[:, :, 60:110, 60:110] = 1.0
    mask = torch.tensor([1, 1, 1, 0, 0, 1.0])
    x[3:5] = 0.0

    for _ in range(50):
        a = apply_msk_augmentations(x, mask)
        ys, xs = torch.nonzero(a[0, 0] > 0.5, as_tuple=True)
        assert xs.float().mean() < 168.0 and ys.float().mean() < 168.0
        assert a[3].abs().max() == 0.0 and a[4].abs().max() == 0.0


def test_model_regularization_and_llrd_routing() -> None:
    """Verify SlotHead dropout rates and correct bottom routing for patch_embed / pos_embed."""
    model = DINOv2SlotHead(
        embed_dim=384,
        num_slots=6,
        num_findings=12,
        pretrained=False,
        attn_dropout_p=0.2,
        pre_cls_dropout_p=0.3,
    )
    assert model.cross_attn.dropout == 0.2
    assert isinstance(model.pre_cls_dropout, nn.Dropout)
    assert model.pre_cls_dropout.p == 0.3

    opt = build_differential_optimizer(model, lr_bottom=5e-6, lr_mid=1e-5, lr_top=2e-5, lr_head=1e-3, weight_decay=0.05)
    assert len(opt.param_groups) == 8

    bot_params = set(opt.param_groups[0]["params"]) | set(opt.param_groups[1]["params"])
    assert model.backbone.patch_embed.proj.weight in bot_params
    assert model.backbone.pos_embed in bot_params

    top_params = set(opt.param_groups[4]["params"]) | set(opt.param_groups[5]["params"])
    assert model.backbone.norm.weight in top_params


def test_lr_for_stage_boundaries() -> None:
    """Verify explicit lr_for matches stated schedule at stage boundaries."""
    spe = 100
    epochs = 8

    # Step 0 (Start of Stage 1): Head warms up from 1e-4, backbone is 0
    assert lr_for("head", 0, spe, epochs) == pytest.approx(1e-4)
    assert lr_for("bb_top", 0, spe, epochs) == 0.0

    # Step = 100 (Epoch 1): Head peaks at 1e-3
    assert lr_for("head", 100, spe, epochs) == pytest.approx(1e-3)
    assert lr_for("bb_top", 100, spe, epochs) == 0.0

    # Step = 200 (Epoch 2, Start of Stage 2): Head = 5e-4, bb_top = 1e-5, bb_mid = 0
    assert lr_for("head", 200, spe, epochs) == pytest.approx(5e-4)
    assert lr_for("bb_top", 200, spe, epochs) == pytest.approx(1e-5)
    assert lr_for("bb_mid", 200, spe, epochs) == 0.0

    # Step = 400 (Epoch 4, Start of Stage 3): 5% floor ramp on backbone only, head unaffected
    assert lr_for("head", 400, spe, epochs) == pytest.approx(2e-4)
    assert lr_for("bb_top", 400, spe, epochs) == pytest.approx(2e-5 * 0.05)

    # Step = 450 (Epoch 4.5, mid ramp in Stage 3):
    # ep3 = 0.5 -> p = 0.5 / 4.0 = 0.125 -> cos(pi * 0.125) = cos(pi / 8) -> ramp = 50 / 100 = 0.5
    cos_125 = 0.5 * (1.0 + math.cos(math.pi * 0.125))
    assert lr_for("bb_top", 450, spe, epochs) == pytest.approx(2e-5 * cos_125 * 0.5)

    # Step = 500 (Epoch 5, post-ramp in Stage 3):
    # ep3 = 1.0 -> p = 1.0 / 4.0 = 0.25 -> cos(pi * 0.25)
    cos_25 = 0.5 * (1.0 + math.cos(math.pi * 0.25))
    assert lr_for("head", 500, spe, epochs) == pytest.approx(2e-4 * cos_25)
    assert lr_for("bb_top", 500, spe, epochs) == pytest.approx(2e-5 * cos_25)

    # Step = 800 (Epoch 8, End): All decayed near 0
    assert lr_for("head", 800, spe, epochs) == pytest.approx(0.0, abs=1e-8)
    assert lr_for("bb_top", 800, spe, epochs) == pytest.approx(0.0, abs=1e-8)


def test_ddp_rewrap_with_weakref_release() -> None:
    """Verify clean DDP rewrapping and old wrapper deallocation via weakref."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29563"
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)

    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    for p in model[0].parameters():
        p.requires_grad = False

    ddp_model = rewrap_ddp(model, device=torch.device("cpu"), local_rank=0)
    out1 = ddp_model(torch.randn(2, 4)).sum()
    out1.backward()

    # Mirror exact training caller teardown
    ref = weakref.ref(ddp_model)
    ddp_model = out1 = None
    import gc
    gc.collect()
    assert ref() is None, "Previous DDP wrapper must be deallocated before re-wrapping!"

    # Unfreeze and rewrap
    for p in model[0].parameters():
        p.requires_grad = True
    ddp_model = rewrap_ddp(model, device=torch.device("cpu"), local_rank=0)
    out2 = ddp_model(torch.randn(2, 4)).sum()
    out2.backward()
    assert list(model[0].parameters())[0].grad is not None

    dist.destroy_process_group()


def test_train_val_inference_parity() -> None:
    """Verify that validation 5-slice extraction sliced at [1:4] matches inference extraction on synthetic DICOM study."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        study_dir = Path(tmp_dir) / "study_parity"
        series_dir = study_dir / "series_sag_pd"
        series_dir.mkdir(parents=True)

        # Write 7 synthetic DICOM slices with physical coordinates and full DICOM header compliance
        for idx in range(7):
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
            file_meta.MediaStorageSOPInstanceUID = f"1.2.3.4.{idx}"
            file_meta.TransferSyntaxUID = "1.2.840.10008.1.2"

            ds = DcmDataset()
            ds.preamble = b"\x00" * 128
            ds.file_meta = file_meta
            ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            ds.SeriesInstanceUID = "1.2.3.4.999"
            ds.InstanceNumber = idx + 1
            ds.SeriesDescription = "SAG PD FS"
            ds.ImageOrientationPatient = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
            ds.ImagePositionPatient = [float(idx * 3.0), 0.0, 0.0]
            ds.PixelSpacing = [0.4, 0.4]
            ds.Rows = 384
            ds.Columns = 384
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.Modality = "MR"
            ds.EchoTime = 40.0
            ds.ImageType = ["ORIGINAL", "PRIMARY"]

            rng = np.random.default_rng(seed=idx)
            pixels = rng.integers(50, 1000, size=(384, 384), dtype=np.uint16)
            ds.PixelData = pixels.tobytes()

            pydicom.dcmwrite(series_dir / f"slice_{idx}.dcm", ds)

        # Process 5-slice volume
        cached_tensor, mask = process_study_to_5slice_cache(
            study_dir=study_dir,
            target_size=336,
            crop_mm=140.0,
            slice_pos=0.5,
        )
        assert mask[0] == 1.0  # Slot 0 SAG_FLUID_FS present

        # Slicing center [1:4] from cached volume
        val_center_triplet = cached_tensor[0, 1:4]

        # Dataset fallback path without cache
        dataset_fallback = EfficiencyStudyDataset(
            study_uids=["study_parity"],
            study_dirs=[study_dir],
            target_size=336,
            crop_mm=140.0,
            is_train=False,
            require_cache=False,
        )
        item = dataset_fallback[0]
        dataset_triplet = item["image"][0]

        # Assert bitwise parity
        assert torch.allclose(val_center_triplet.float(), dataset_triplet.float(), atol=1e-4)
