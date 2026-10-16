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

import json
import math
import os
from pathlib import Path
import random
import tempfile
import weakref

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset as DcmDataset, FileMetaDataset
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from src.datasets.efficiency_pipeline import (
    select_5slice_paths,
    normalize_triplet_dinov2,
    crop_and_resize_140mm,
    process_study_to_5slice_cache,
)
from src.models.dinov2_slothead import DINOv2SlotHead
from src.training.losses import MaskedBCEWithLogitsLoss
from src.training.train_efficiency import (
    STAGE4_LR_PEAKS,
    EfficiencyStudyDataset,
    apply_msk_augmentations,
    build_differential_optimizer,
    compute_collective_val_loss,
    impute_soft_synovitis,
    stage_for_epoch,
    stage4_start_for,
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
        assert torch.allclose(val_item["image"], tensor_5[:, 1:4].float())

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


def test_stage_for_epoch_and_lr_alignment() -> None:
    """Verify single source of truth for unfreezing stage boundaries and peaks."""
    assert [stage_for_epoch(e, total_epochs=10) for e in range(1, 11)] == [1, 1, 2, 2, 3, 3, 4, 4, 4, 4]
    assert [stage_for_epoch(e, total_epochs=14) for e in range(1, 15)] == [1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4]
    assert set(STAGE4_LR_PEAKS.keys()) == {"bb_bot", "bb_mid", "bb_top1", "bb_top2", "head"}


def test_stage_coupling_lr_and_unfreezing() -> None:
    """Verify that in every stage, families with lr_for > 0 match requires_grad parameters exactly."""
    model = DINOv2SlotHead(pretrained=False)
    spe = 100
    epochs = 10

    expected_active_families = {
        1: {"head"},
        2: {"head", "bb_top2"},
        3: {"head", "bb_top2", "bb_top1"},
        4: {"head", "bb_top2", "bb_top1", "bb_mid", "bb_bot"},
    }

    for stage in (1, 2, 3, 4):
        apply_unfreezing_stage(model, stage)
        step = (2 * stage - 1) * spe  # Middle of stage

        active_lr_families = {
            fam for fam in ("head", "bb_top2", "bb_top1", "bb_mid", "bb_bot")
            if lr_for(fam, step, spe, epochs) > 0.0
        }
        assert active_lr_families == expected_active_families[stage]


def test_apply_unfreezing_stages_four_stage_progression() -> None:
    """Verify parameter requires_grad transitions across all 4 stages."""
    model = DINOv2SlotHead(pretrained=False)

    # Stage 1: Entire backbone frozen
    apply_unfreezing_stage(model, 1)
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert model.finding_queries.requires_grad
    assert model.slot_embed.requires_grad
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    assert all(p.requires_grad for p in head_params)

    # Stage 2: Top-2 blocks (10, 11) + norm unfrozen
    apply_unfreezing_stage(model, 2)
    assert all(p.requires_grad for p in model.backbone.blocks[10].parameters())
    assert all(p.requires_grad for p in model.backbone.blocks[11].parameters())
    assert all(p.requires_grad for p in model.backbone.norm.parameters())
    assert not any(p.requires_grad for p in model.backbone.blocks[9].parameters())
    assert not any(p.requires_grad for p in model.backbone.patch_embed.parameters())

    # Stage 3: Top-4 blocks (8, 9, 10, 11) + norm unfrozen
    apply_unfreezing_stage(model, 3)
    assert all(p.requires_grad for p in model.backbone.blocks[8].parameters())
    assert all(p.requires_grad for p in model.backbone.blocks[9].parameters())
    assert all(p.requires_grad for p in model.backbone.blocks[10].parameters())
    assert all(p.requires_grad for p in model.backbone.blocks[11].parameters())
    assert all(p.requires_grad for p in model.backbone.norm.parameters())
    assert not any(p.requires_grad for p in model.backbone.blocks[7].parameters())

    # Stage 4: Entire backbone unfrozen
    apply_unfreezing_stage(model, 4)
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_optimizer_ten_param_groups_llrd_routing() -> None:
    """Verify build_differential_optimizer creates 10 groups without dead shims."""
    model = DINOv2SlotHead(pretrained=False)
    opt = build_differential_optimizer(
        model,
        lr_bottom=5e-6,
        lr_mid=1e-5,
        lr_top1=1.5e-5,
        lr_top2=2e-5,
        lr_head=1e-3,
        weight_decay=0.05,
    )
    assert len(opt.param_groups) == 10

    group_names = [g["name"] for g in opt.param_groups]
    expected_names = [
        "bb_bot_decay", "bb_bot_nodecay",
        "bb_mid_decay", "bb_mid_nodecay",
        "bb_top1_decay", "bb_top1_nodecay",
        "bb_top2_decay", "bb_top2_nodecay",
        "head_decay", "head_nodecay",
    ]
    assert group_names == expected_names

    bot_params = set(opt.param_groups[0]["params"]) | set(opt.param_groups[1]["params"])
    assert model.backbone.patch_embed.proj.weight in bot_params

    mid_params = set(opt.param_groups[2]["params"]) | set(opt.param_groups[3]["params"])
    assert model.backbone.blocks[5].attn.qkv.weight in mid_params

    top1_params = set(opt.param_groups[4]["params"]) | set(opt.param_groups[5]["params"])
    assert model.backbone.blocks[8].attn.qkv.weight in top1_params
    assert model.backbone.blocks[9].attn.qkv.weight in top1_params

    top2_params = set(opt.param_groups[6]["params"]) | set(opt.param_groups[7]["params"])
    assert model.backbone.blocks[10].attn.qkv.weight in top2_params
    assert model.backbone.norm.weight in top2_params


def test_lr_for_schedule_ramps_and_continuity() -> None:
    """Verify explicit lr_for matches stated schedule and avoids downward spikes for warm layers."""
    spe = 100
    epochs = 10

    # Stage 1: Head warmup & decay
    assert lr_for("head", 0, spe, epochs) == pytest.approx(1e-4)
    assert lr_for("bb_top2", 0, spe, epochs) == 0.0
    assert lr_for("head", 100, spe, epochs) == pytest.approx(1e-3)

    # Stage 2 start (step 200): bb_top2 warms up with 100-step ramp
    assert lr_for("head", 200, spe, epochs) == pytest.approx(5e-4)
    assert lr_for("bb_top2", 200, spe, epochs) == pytest.approx(1e-5 * 0.05)
    assert lr_for("bb_top2", 300, spe, epochs) == pytest.approx(1e-5)
    assert lr_for("bb_top1", 250, spe, epochs) == 0.0

    # Stage 3 start (step 400): bb_top2 steps up cleanly; bb_top1 starts ramp
    assert lr_for("head", 400, spe, epochs) == pytest.approx(2e-4)
    assert lr_for("bb_top2", 400, spe, epochs) == pytest.approx(1.5e-5)
    assert lr_for("bb_top1", 400, spe, epochs) == pytest.approx(1e-5 * 0.05)
    assert lr_for("bb_mid", 450, spe, epochs) == 0.0

    # Stage 4 start (step 600): warm layers step up to cosine peak; newly unfrozen ramp
    assert lr_for("head", 600, spe, epochs) == pytest.approx(2e-4)
    assert lr_for("bb_top2", 600, spe, epochs) == pytest.approx(2e-5)       # Clean step, no plunge
    assert lr_for("bb_top1", 600, spe, epochs) == pytest.approx(1.5e-5)     # Clean step, no plunge
    assert lr_for("bb_mid", 600, spe, epochs) == pytest.approx(1e-5 * 0.05) # Ramped
    assert lr_for("bb_bot", 600, spe, epochs) == pytest.approx(5e-6 * 0.05) # Ramped

    # Stage 4 post-ramp (step 700): 100 steps in, all layers on cosine curve
    cos_25 = 0.5 * (1.0 + math.cos(math.pi * 0.25))
    assert lr_for("bb_top2", 700, spe, epochs) == pytest.approx(2e-5 * cos_25)
    assert lr_for("bb_mid", 700, spe, epochs) == pytest.approx(1e-5 * cos_25)

    # End (step 1000): Decayed to zero
    assert lr_for("head", 1000, spe, epochs) == pytest.approx(0.0, abs=1e-8)


def test_soft_supervision_silent_only_routing() -> None:
    """Verify that impute_soft_synovitis applies strictly to silent non-gold rows."""
    df = pd.DataFrame({
        "StudyInstanceUID": ["u1", "u2", "u3", "u4"],
        "is_gold": [0, 0, 0, 1],
        "Synovitis": [np.nan, 1.0, 0.0, np.nan],
        "synovitis_soft": [0.63, 0.22, 0.63, 0.22],
    })
    label_cols = ["Synovitis"]

    n = impute_soft_synovitis(df, label_cols)
    assert n == 1
    assert df.loc[0, "Synovitis"] == 0.63       # Non-gold silent row imputed
    assert df.loc[1, "Synovitis"] == 1.0        # Non-gold explicit positive PRESERVED
    assert df.loc[2, "Synovitis"] == 0.0        # Non-gold explicit negative PRESERVED
    assert np.isnan(df.loc[3, "Synovitis"])     # Gold row NEVER modified


def test_soft_supervision_pre_binarized_routing() -> None:
    """Verify that impute_soft_synovitis works on pre-binarized CSV (e.g. pseudo_labels.csv)."""
    df = pd.DataFrame({
        "StudyInstanceUID": ["u1", "u2", "u3", "u4"],
        "is_gold": [0, 0, 0, 1],
        "Synovitis": [0.0, 1.0, 0.0, 0.0],
        "synovitis_soft": [0.63, 1.0, 0.0, np.nan],
    })
    label_cols = ["Synovitis"]

    n = impute_soft_synovitis(df, label_cols)
    assert n == 1
    assert df.loc[0, "Synovitis"] == 0.63       # Non-gold silent row (mapped to 0.0) receives soft label
    assert df.loc[1, "Synovitis"] == 1.0        # Non-gold explicit positive PRESERVED
    assert df.loc[2, "Synovitis"] == 0.0        # Non-gold explicit negative PRESERVED
    assert df.loc[3, "Synovitis"] == 0.0        # Gold row NEVER modified

    # Idempotent second invocation
    n_second = impute_soft_synovitis(df, label_cols)
    assert n_second == 1
    assert df.loc[0, "Synovitis"] == 0.63


def test_checkpoint_eligibility_stage4_restriction() -> None:
    """Verify stage4_start_for helper strictly enforces Stage 4 threshold."""
    assert stage4_start_for(10) == 7
    assert stage4_start_for(8) == 7
    assert stage4_start_for(6) == 1
    assert stage4_start_for(4) == 1


def test_collective_validation_loss_device_and_finite() -> None:
    """Verify compute_collective_val_loss computes finite loss on CPU and asserts finite output."""
    device = torch.device("cpu")
    pos_weights = torch.ones(12, dtype=torch.float32, device=device)
    criterion = MaskedBCEWithLogitsLoss(pos_weight=pos_weights)

    val_records = [
        {"study_id": f"s_{i}", "logits": [0.5] * 12, "targets": [1.0] * 12, "is_gold": 0}
        for i in range(4)
    ]
    val_logits_np = np.array([r["logits"] for r in val_records])
    val_targets_np = np.array([r["targets"] for r in val_records])
    val_is_gold_np = np.array([r["is_gold"] for r in val_records])

    loss = compute_collective_val_loss(val_logits_np, val_targets_np, val_is_gold_np, criterion, device)
    assert math.isfinite(loss)
    assert loss > 0.0


def test_collective_validation_loss_raises_on_non_finite() -> None:
    """Verify compute_collective_val_loss raises RuntimeError when loss is non-finite."""
    device = torch.device("cpu")
    val_logits_np = np.array([[float("inf")] * 12])
    val_targets_np = np.array([[1.0] * 12])
    val_is_gold_np = np.array([0])

    with pytest.raises(RuntimeError, match="non-finite"):
        compute_collective_val_loss(val_logits_np, val_targets_np, val_is_gold_np, criterion=None, device=device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU loss validation")
def test_collective_validation_loss_gpu_device_match() -> None:
    """Verify compute_collective_val_loss runs on CUDA device without tensor mismatch."""
    device = torch.device("cuda:0")
    pos_weights = torch.ones(12, dtype=torch.float32, device=device)
    criterion = MaskedBCEWithLogitsLoss(pos_weight=pos_weights)

    val_logits_np = np.ones((2, 12), dtype=np.float32) * 0.5
    val_targets_np = np.ones((2, 12), dtype=np.float32)
    val_is_gold_np = np.zeros(2, dtype=np.int64)

    loss = compute_collective_val_loss(val_logits_np, val_targets_np, val_is_gold_np, criterion, device)
    assert math.isfinite(loss)


def test_gradient_checkpointing_partial_unfreezing() -> None:
    """Verify that under gradient checkpointing in Stage 2, unfrozen blocks receive non-None gradients."""
    model = DINOv2SlotHead(pretrained=False)
    if hasattr(model.backbone, "set_grad_checkpointing"):
        model.backbone.set_grad_checkpointing(True)
    apply_unfreezing_stage(model, stage=2)

    imgs = torch.randn(2, 6, 3, 336, 336)
    masks = torch.ones(2, 6)
    logits = model(imgs, masks)
    loss = logits.sum()
    loss.backward()

    # Frozen early blocks must have None grad
    assert model.backbone.blocks[0].norm1.weight.grad is None
    # Unfrozen top blocks and SlotHead queries must have valid non-None grads
    assert model.backbone.blocks[10].norm1.weight.grad is not None
    assert model.backbone.blocks[11].norm1.weight.grad is not None
    assert model.finding_queries.grad is not None


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


def test_dinov2_slothead_336px_shape_contract() -> None:
    """Verify DINOv2SlotHead instantiates with exact (1, 577, 384) pos_embed shape at 336px."""
    model = DINOv2SlotHead(img_size=336, pretrained=False)
    # 1 CLS token + (336 // 14)^2 = 1 + 24*24 = 577 tokens
    assert model.backbone.pos_embed.shape == (1, 577, 384)

    x = torch.randn(1, 6, 3, 336, 336)
    mask = torch.ones(1, 6)
    with torch.no_grad():
        out = model(x, mask)
    assert out.shape == (1, 12)
    assert torch.isfinite(out).all()


def test_dinov2_slothead_forward_backbone_eval_parity() -> None:
    """Verify _forward_backbone block loop matches backbone(x) exactly in eval mode."""
    model = DINOv2SlotHead(img_size=336, pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 336, 336)
    with torch.no_grad():
        out_custom = model._forward_backbone(x)
        out_timm = model.backbone(x)

    assert torch.allclose(out_custom, out_timm, atol=1e-5)


def test_stage2_grads_reach_top_blocks_only() -> None:
    """CPU test verifying Stage 2 non-reentrant checkpointing allows gradients to reach blocks 10-11."""
    from src.training.ddp import set_stage_trainable

    model = DINOv2SlotHead(img_size=56, pretrained=False)
    model.train()
    set_stage_trainable(model, stage=2)

    x = torch.randn(2, 6, 3, 56, 56)
    mask = torch.ones(2, 6)
    out = model(x, mask)
    out.sum().backward()

    # Early blocks must have None grad
    assert all(p.grad is None for p in model.backbone.blocks[0].parameters())
    # Unfrozen top blocks must have non-None gradients
    assert all(p.grad is not None for p in model.backbone.blocks[11].parameters())


@pytest.mark.skipif("RSNA_BACKBONE_PATH" not in os.environ, reason="Requires offline backbone checkpoint")
def test_dinov2_slothead_336px_pretrained_resampling() -> None:
    """Verify offline checkpoint loads with timm resampled pos_embed matching reference resampler."""
    from timm.layers import resample_abs_pos_embed

    weights_path = os.environ["RSNA_BACKBONE_PATH"]
    model = DINOv2SlotHead(img_size=336, pretrained=False, weights_path=weights_path)
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "model" in sd:
        sd = sd["model"]
    elif "state_dict" in sd:
        sd = sd["state_dict"]
    elif "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    assert "pos_embed" in sd, f"Checkpoint has no pos_embed: keys {list(sd.keys())[:5]}"
    ref = resample_abs_pos_embed(sd["pos_embed"], new_size=(24, 24), num_prefix_tokens=1)
    assert torch.allclose(model.backbone.pos_embed, ref, atol=1e-4)


def test_continuous_soft_supervision_loss_with_random_logits() -> None:
    """Verify MaskedBCEWithLogitsLoss with random non-zero logits matches PyTorch manual reduction."""
    torch.manual_seed(123)
    gold_weight = 2.0
    loss_fn = MaskedBCEWithLogitsLoss(gold_weight=gold_weight)

    logits = torch.randn(2, 12)
    targets = torch.tensor([
        [0.63, 0.22, 0.10, 0.90, 0.45, 0.05, 0.80, 0.15, 0.50, 0.70, 0.30, 0.60],
        [1.00, 0.00, 1.00, 0.00, 1.00, 0.00, 1.00, 0.00, 1.00, 0.00, 1.00, 0.00],
    ], dtype=torch.float32)
    is_gold = torch.tensor([0, 1], dtype=torch.long)

    loss = loss_fn(logits, targets, is_gold)
    assert math.isfinite(loss.item())

    raw_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    row_weights = torch.tensor([[1.0], [gold_weight]])
    expected_loss = (raw_bce * row_weights).sum() / 24.0

    assert torch.allclose(loss, expected_loss, atol=1e-5)


def test_lr_for_schedule_boundary_values() -> None:
    """Verify unified lr_for schedule boundary mathematical invariants for 14-epoch training."""
    spe = 100
    epochs = 14

    # 1. Stage 1 end (epoch 2.0): Head warm-up reaches 5e-4, backbone is 0.0
    lr_head_s1_end = lr_for("head", step=2 * spe, spe=spe, epochs=epochs)
    assert abs(lr_head_s1_end - 5e-4) < 1e-6
    assert lr_for("bb_top2", step=2 * spe - 1, spe=spe, epochs=epochs) == 0.0

    # 2. Stage 3 (epoch 5.5): bb_top2 is exactly 1.5e-5, head is 2e-4
    lr_top2_s3 = lr_for("bb_top2", step=int(5.5 * spe), spe=spe, epochs=epochs)
    lr_head_s3 = lr_for("head", step=int(5.5 * spe), spe=spe, epochs=epochs)
    assert abs(lr_top2_s3 - 1.5e-5) < 1e-6
    assert abs(lr_head_s3 - 2e-4) < 1e-6

    # 3. Stage 4 final step: All learning rates anneal cleanly to ~0.0
    final_step = 14 * spe
    for fam in ("head", "bb_top2", "bb_top1", "bb_mid", "bb_bot"):
        lr_final = lr_for(fam, step=final_step, spe=spe, epochs=epochs)
        assert lr_final < 1e-6, f"{fam} did not anneal to 0: {lr_final}"


def test_14_epoch_schedule_stage_mapping() -> None:
    """Verify stage_for_epoch and stage4_start_for accurately reflect the 14-epoch schedule."""
    assert stage_for_epoch(1, 14) == 1
    assert stage_for_epoch(2, 14) == 1
    assert stage_for_epoch(3, 14) == 2
    assert stage_for_epoch(4, 14) == 2
    assert stage_for_epoch(5, 14) == 3
    assert stage_for_epoch(6, 14) == 3
    assert stage_for_epoch(7, 14) == 3
    for ep in range(8, 15):
        assert stage_for_epoch(ep, 14) == 4
    assert stage4_start_for(14) == 8


def test_crop_and_resize_tight_fov_padding() -> None:
    """Verify that a non-square 100mm x 128mm FOV image is symmetrically padded to square before resizing, preserving physical scale."""
    from src.datasets.efficiency_pipeline import crop_and_resize_140mm

    # 250 x 320 array with 0.4 mm spacing -> 100 mm x 128 mm FOV
    vol = np.zeros((5, 250, 320), dtype=np.float32)
    # 20 mm square feature at 0.4 mm spacing: 50 x 50 pixels centered
    vol[:, 100:150, 135:185] = 2.0

    out, info = crop_and_resize_140mm(vol, pixel_spacing=(0.4, 0.4), target_size=336, crop_mm=140.0)
    assert out.shape == (5, 336, 336)
    assert info["clipped"]

    # In 140 mm crop resized to 336 px, 20 mm feature must span:
    # 20 / 140 * 336 = 48.0 pixels across center row
    feature_span_x = int((out[0, 168] > 1.0).sum())
    assert abs(feature_span_x - 48) <= 2, f"Expected ~48 px span, got {feature_span_x}"


def test_efficiency_dataset_npz_loading(tmp_path: Path) -> None:
    """Verify EfficiencyStudyDataset seamlessly loads compressed .npz archives."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    uid = "test_npz_study"

    img_data = np.random.randn(6, 5, 336, 336).astype(np.float16)
    mask_data = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    with open(cache_dir / f"{uid}.npz", "wb") as f:
        np.savez_compressed(f, image=img_data, presence_mask=mask_data, clipped=np.uint8(0))

    ds = EfficiencyStudyDataset(
        study_uids=[uid],
        study_dirs=[tmp_path],
        labels=np.zeros((1, 12), dtype=np.float32),
        is_gold=np.zeros(1, dtype=np.int64),
        cached_tensor_dir=cache_dir,
        target_size=336,
        is_train=False,
        require_cache=True,
    )

    item = ds[0]
    assert item["study_id"] == uid
    assert item["image"].shape == (6, 3, 336, 336)
    assert item["presence_mask"].shape == (6,)
    assert torch.allclose(item["presence_mask"], torch.from_numpy(mask_data))


def test_verify_cache_manifest_and_files(tmp_path: Path) -> None:
    """Verify verify_cache validates manifest metadata and reports missing files."""
    from src.datasets.efficiency_pipeline import verify_cache
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    manifest = {
        "target_size": 336,
        "crop_mm": 140.0,
        "n_slices_cached": 5,
        "format": "npz",
        "in_progress": False,
        "failed_uids": [],
    }
    with open(cache_dir / "cache_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    (cache_dir / "study_001.npz").touch()
    with pytest.raises(RuntimeError, match="study_002"):
        verify_cache(cache_dir, ["study_001", "study_002"], target_size=336, crop_mm=140.0)

    # In-progress check
    manifest["in_progress"] = True
    with open(cache_dir / "cache_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with pytest.raises(RuntimeError, match="incomplete"):
        verify_cache(cache_dir, ["study_001"], target_size=336, crop_mm=140.0)

    # Size mismatch check
    manifest["in_progress"] = False
    manifest["target_size"] = 224
    with open(cache_dir / "cache_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        verify_cache(cache_dir, ["study_001"], target_size=336, crop_mm=140.0)
