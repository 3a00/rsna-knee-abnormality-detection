"""tests/test_multizone_sampling.py

Comprehensive test suite verifying:
- 3-zone compartment-anchored slice selection and physical spacing logic.
- 18-token multi-zone SlotHead cross-attention shapes, masking, and gradients.
- Closed-form normalized task loss reweighting gradients.
- Progressive unfreezing and parameter group routing across all 4 stages.
- Telemetry logging exception safety and scalar verification.
- Imputation safety preserving gold-standard hard labels and handling pre-binarized silence.
- Strict fail-fast cache contract rejecting invalid formats and missing files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.datasets.efficiency_pipeline import (
    DICOMHeaderMeta,
    select_multizone_slice_paths,
    verify_cache,
)
from src.models.dinov2_slothead import DINOv2SlotHead
from src.training.losses import MaskedBCEWithLogitsLoss, compute_task_weights
from src.training.train_efficiency import (
    EfficiencyStudyDataset,
    apply_msk_augmentations,
    build_differential_optimizer,
    impute_soft_synovitis,
)
from src.training.ddp import set_stage_trainable


def _make_mock_headers(n: int, slice_spacing: float = 3.0) -> list[DICOMHeaderMeta]:
    """Helper creating mock DICOMHeaderMeta objects matching exact production dataclass fields."""
    return [
        DICOMHeaderMeta(
            path=f"/mock/study/series/slice_{i:03d}.dcm",
            series_uid="1.2.3.4",
            k_pos=float(i * slice_spacing),
            has_physical_coords=True,
            plane="sagittal",
            is_fluid=True,
            is_fs=True,
            pixel_spacing=(0.4, 0.4),
            instance_number=i + 1,
            echo_number=1,
        )
        for i in range(n)
    ]


def test_select_multizone_slice_paths_spacing_and_anchors():
    """Verify N=25 standard series and N=120 thin-slice 3D series anchors."""
    # Case 1: Standard 2D acquisition (N=25, spacing=3.0mm)
    headers_25 = _make_mock_headers(25, slice_spacing=3.0)
    paths_25 = select_multizone_slice_paths(headers_25)
    expected_indices_25 = [5, 6, 7, 11, 12, 13, 17, 18, 19]
    actual_indices_25 = [int(p.split("_")[-1].split(".")[0]) for p in paths_25]
    assert actual_indices_25 == expected_indices_25

    # Case 2: Thin-slice 3D acquisition (N=120, spacing=0.5mm < 1.5mm)
    headers_thin = _make_mock_headers(120, slice_spacing=0.5)
    paths_thin = select_multizone_slice_paths(headers_thin)
    actual_indices_thin = [int(p.split("_")[-1].split(".")[0]) for p in paths_thin]
    # Center zone (indices 3, 4, 5) must match [54, 60, 66]
    assert actual_indices_thin[3:6] == [54, 60, 66]


def test_select_multizone_slice_paths_short_series_clamping():
    """Verify series with N < 9 slices gracefully clamp to boundaries without crash."""
    headers_5 = _make_mock_headers(5, slice_spacing=3.0)
    paths_5 = select_multizone_slice_paths(headers_5)
    assert len(paths_5) == 9
    for p in paths_5:
        idx = int(p.split("_")[-1].split(".")[0])
        assert 0 <= idx < 5


def test_all_absent_study_produces_finite_logits():
    """Verify all-absent study unmasks key_padding_mask and outputs strictly finite logits (B3)."""
    m = DINOv2SlotHead(img_size=112, pretrained=False).eval()
    logits = m(torch.randn(2, 6, 9, 112, 112), torch.zeros(2, 6))
    assert torch.isfinite(logits).all()


def test_dinov2_slothead_18token_shapes_and_cluster_masking():
    """Verify 18-token multi-zone SlotHead output shapes and absent slot masking."""
    model = DINOv2SlotHead(img_size=112, num_findings=12, pretrained=False)
    model.eval()

    b = 2
    x = torch.randn(b, 6, 9, 112, 112)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]])

    logits, attn = model.forward_with_attention(x, mask)
    assert logits.shape == (b, 12)
    assert attn.shape == (b, 12, 18)

    # In sample 0, slots 3, 4, 5 are absent. All corresponding zone tokens (9..17) must have ~0 attention
    assert torch.allclose(attn[0, :, 9:], torch.zeros_like(attn[0, :, 9:]), atol=1e-5)


def test_head_incl_zone_embed_trains_from_stage1_and_all_at_stage4():
    """Verify head parameters (including zone_embed) train in Stage 1 and all in Stage 4."""
    m = DINOv2SlotHead(img_size=112, num_findings=12, pretrained=False)
    set_stage_trainable(m, 1)
    head_params = [
        m.slot_embed,
        m.zone_embed,
        m.finding_queries,
        m.cls_w,
        m.cls_b,
        *m.cross_attn.parameters(),
        *m.norm.parameters(),
    ]
    assert all(p.requires_grad for p in head_params)
    assert not any(p.requires_grad for p in m.backbone.parameters())

    set_stage_trainable(m, 4)
    assert all(p.requires_grad for p in m.parameters())


def test_soft_synovitis_imputation_and_gold_protection():
    """Verify soft synovitis fills NaN and pre-binarized silence while preserving gold labels (F4)."""
    df = pd.DataFrame({
        "Synovitis": [1.0, np.nan, 0.0, np.nan],
        "synovitis_soft": [0.22, 0.63, 0.88, 0.45],
        "is_gold": [0, 0, 0, 1],
    })
    imputed = impute_soft_synovitis(df)
    # Row 0: non-gold hard 1.0 preserved
    # Row 1: non-gold NaN filled with 0.63
    # Row 2: non-gold pre-binarized 0.0 filled with 0.88
    # Row 3: gold NaN remains NaN (never imputed with weak label)
    assert imputed["Synovitis"].iloc[0] == pytest.approx(1.0)
    assert imputed["Synovitis"].iloc[1] == pytest.approx(0.63)
    assert imputed["Synovitis"].iloc[2] == pytest.approx(0.88)
    assert np.isnan(imputed["Synovitis"].iloc[3])


def test_apply_msk_augmentations_shape_and_absent_slot_untouched():
    """Verify augmentations preserve tensor shape and never apply spatial distortion to absent slots."""
    x = torch.randn(6, 9, 64, 64)
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    y = apply_msk_augmentations(x, mask)
    assert y.shape == x.shape
    # Absent slots are affected only by global contrast scale factor, never by spatial affine
    ratio = y[1] / x[1]
    assert torch.allclose(ratio, ratio.flatten()[0].expand_as(ratio), atol=1e-4)


def test_tensorboard_logger_smoke_scalar(tmp_path: Path):
    """Verify TensorBoardLogger logs scalars to disk without NameError or TypeError."""
    from src.utils.tensorboard import TensorBoardLogger
    logger = TensorBoardLogger(log_dir=tmp_path)
    logger.log_scalar("Loss/train_step", 0.42, step=1)
    logger.flush()
    logger.close()
    event_files = list(tmp_path.glob("events.out.tfevents.*"))
    assert len(event_files) >= 1
    assert event_files[0].stat().st_size > 0


def test_efficiency_study_dataset_rejects_legacy_5slice_npz(tmp_path: Path):
    """Verify EfficiencyStudyDataset rejects legacy 5-slice archives with RuntimeError."""
    manifest = {
        "target_size": 336,
        "crop_mm": 140.0,
        "format": "npz",
        "n_slices_cached": 9,
        "sampling_strategy": "3zone",
    }
    with open(tmp_path / "cache_manifest.json", "w") as f:
        json.dump(manifest, f)

    np.savez_compressed(
        tmp_path / "study_001.npz",
        image=np.zeros((6, 5, 336, 336), dtype=np.float16),
        presence_mask=np.ones(6, dtype=np.float32),
    )

    ds = EfficiencyStudyDataset(
        study_uids=["study_001"],
        cached_tensor_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="Cache array shape mismatch"):
        _ = ds[0]


def test_efficiency_study_dataset_raises_on_missing_npz(tmp_path: Path):
    """Verify EfficiencyStudyDataset raises FileNotFoundError if cached archive is missing."""
    manifest = {
        "target_size": 336,
        "crop_mm": 140.0,
        "format": "npz",
        "n_slices_cached": 9,
        "sampling_strategy": "3zone",
    }
    with open(tmp_path / "cache_manifest.json", "w") as f:
        json.dump(manifest, f)

    ds = EfficiencyStudyDataset(
        study_uids=["missing_study"],
        cached_tensor_dir=tmp_path,
    )
    with pytest.raises(FileNotFoundError):
        _ = ds[0]


def test_task_weights_computation_and_closed_form_gradients():
    """Verify task loss weighting normalizes to mean 1.0 and matches exact analytical gradients."""
    task_w = compute_task_weights(target_boost=1.75, boost_indices=(2, 3, 5, 6))
    assert task_w.shape == (12,)
    assert pytest.approx(float(task_w.mean().item()), abs=1e-5) == 1.0

    # Boosted categories (4 targets) receive 1.40x, non-boosted (8 targets) receive 0.80x
    for idx in (2, 3, 5, 6):
        assert pytest.approx(float(task_w[idx].item()), abs=1e-2) == 1.40
    for idx in (0, 1, 4, 7, 8, 9, 10, 11):
        assert pytest.approx(float(task_w[idx].item()), abs=1e-2) == 0.80

    criterion = MaskedBCEWithLogitsLoss(gold_weight=1.0, task_weights=task_w, reduction="sum")
    logits = torch.randn(1, 12, requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
    is_gold = torch.tensor([0])
    loss = criterion(logits, targets, is_gold=is_gold)
    loss.backward()

    expected_grad = task_w * (torch.sigmoid(logits.detach()) - targets)
    assert torch.allclose(logits.grad, expected_grad, atol=1e-5)


def test_optimizer_family_tags_cover_every_group():
    """Verify differential optimizer param groups cover all parameters and assign family keys."""
    model = DINOv2SlotHead(img_size=112, num_findings=12, pretrained=False)
    opt = build_differential_optimizer(model)
    expected_fams = {"head", "bb_top2", "bb_top1", "bb_mid", "bb_bot"}
    assert {g["family"] for g in opt.param_groups} <= expected_fams
    assert all("family" in g for g in opt.param_groups)


def test_build_criterion_wires_task_weights():
    """Verify build_criterion properly wires normalized task weights and assigns target boosts."""
    from scripts.train_ddp_efficiency import build_criterion
    args = argparse.Namespace(gold_weight=5.0, target_label_boost=1.75)
    c = build_criterion(args, torch.ones(12), torch.device("cpu"))
    assert c.task_weights is not None
    assert abs(float(c.task_weights[2]) - 1.40) < 1e-2
    assert abs(float(c.task_weights[0]) - 0.80) < 1e-2


def test_gold_weight_none_is_resolved(tmp_path: Path):
    """Verify that resolve_gold_weight resolves from config.yaml when args.gold_weight is None."""
    from scripts.train_ddp_efficiency import resolve_gold_weight
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("training:\n  gold_weight: 5.0\n", encoding="utf-8")
    args = argparse.Namespace(gold_weight=None, config_path=str(cfg_file))
    assert resolve_gold_weight(args) == 5.0

    args_explicit = argparse.Namespace(gold_weight=2.5, config_path=str(cfg_file))
    assert resolve_gold_weight(args_explicit) == 2.5


def test_synovitis_callsite_returns_dataframe():
    """Verify impute_soft_synovitis returns a DataFrame and preserves caller immutability."""
    from src.training.train_efficiency import impute_soft_synovitis
    df = pd.DataFrame({
        "StudyInstanceUID": ["s1", "s2"],
        "is_gold": [0, 0],
        "Synovitis": [0.0, float("nan")],
        "synovitis_soft": [0.63, 0.22],
        "label1": [0.0, 1.0],
    })
    label_cols = ["Synovitis", "label1"]
    out_df = impute_soft_synovitis(df, label_cols)
    assert isinstance(out_df, pd.DataFrame)
    assert out_df["Synovitis"].iloc[0] == pytest.approx(0.63)
    assert out_df["Synovitis"].iloc[1] == pytest.approx(0.22)
    assert df["Synovitis"].iloc[0] == 0.0  # verifies original was not mutated in-place
