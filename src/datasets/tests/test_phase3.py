"""
test_phase3.py -- Phase 3 Unit Tests (v3 Final)

Covers all major components of the Phase 3 image baseline:
    - select_slices: shape, padding, dtype
    - stack_25d: shape, boundary clamping, center channel identity, even-size guard
    - 2.5D augmentation alignment: stack-before-augment order preserved
    - synovitis_soft wiring: non-gold uses soft target, gold uses hard label
    - Bilateral study: keep-all strategy (no series deleted, warning emitted)
    - KneeMILModel: forward shape, raw logits (no sigmoid), attention sums, plane mask
    - AttentionPooling: output shape, softmax constraint
    - MaskedBCEWithLogitsLoss: finite loss, NaN masking, gold upweighting
    - compute_per_label_auc: perfect AUC, NaN exclusion, macro averaging

Run with:
    pytest src/datasets/tests/test_phase3.py -v
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from src.datasets.mri_dataset import (
    KneeMRIDataset,
    select_slices,
    stack_25d,
    SYNOVITIS_COL,
    SYNOVITIS_SOFT_COL,
)
from src.models.mil_model import AttentionPooling, KneeMILModel
from src.training.losses import MaskedBCEWithLogitsLoss
from src.training.trainer import compute_per_label_auc


LABEL_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def dummy_volume() -> np.ndarray:
    return np.random.default_rng(42).random((30, 64, 64)).astype(np.float32)


@pytest.fixture
def dummy_model() -> KneeMILModel:
    return KneeMILModel(
        backbone_name="efficientnet_b0",
        n_classes=12,
        stack_size=3,
        pretrained=False,
        use_grad_checkpointing=False,  # Disable for unit tests (no GPU needed)
    )


@pytest.fixture
def dummy_planes() -> dict[str, torch.Tensor]:
    B, N, C, H, W = 2, 4, 3, 64, 64
    return {p: torch.randn(B, N, C, H, W) for p in ["sagittal", "coronal", "axial"]}


# ============================================================================
# Test: select_slices
# ============================================================================

class TestSelectSlices:
    def test_output_shape_larger_volume(self, dummy_volume):
        result = select_slices(dummy_volume, n_slices=24, seed=42)
        assert result.shape == (24, 64, 64)

    def test_padding_for_small_volume(self):
        small = np.ones((10, 16, 16), dtype=np.float32)
        out = select_slices(small, n_slices=24, seed=42)
        assert out.shape == (24, 16, 16)
        assert np.all(out[10:] == 0.0), "Padded slices must be zeros"

    def test_exact_size_volume(self):
        exact = np.ones((24, 16, 16), dtype=np.float32)
        out = select_slices(exact, n_slices=24, seed=42)
        assert out.shape == (24, 16, 16)

    def test_dtype_preserved_float32(self, dummy_volume):
        assert select_slices(dummy_volume, n_slices=16, seed=42).dtype == np.float32


# ============================================================================
# Test: stack_25d
# ============================================================================

class TestStack25D:
    def test_output_shape(self):
        vol = np.ones((24, 16, 16), dtype=np.float32)
        assert stack_25d(vol, 3).shape == (24, 3, 16, 16)

    def test_boundary_clamping_first_slice(self):
        vol = np.random.rand(10, 16, 16).astype(np.float32)
        stacked = stack_25d(vol, 3)
        # At slice 0: prev channel (offset=-1) clamped to 0 -> same as current
        assert np.allclose(stacked[0, 0], stacked[0, 1])

    def test_boundary_clamping_last_slice(self):
        vol = np.random.rand(10, 16, 16).astype(np.float32)
        stacked = stack_25d(vol, 3)
        # At last slice: next channel (offset=+1) clamped -> same as current
        assert np.allclose(stacked[-1, 2], stacked[-1, 1])

    def test_center_channel_matches_original(self):
        vol = np.random.default_rng(0).random((20, 16, 16)).astype(np.float32)
        stacked = stack_25d(vol, 3)
        for i in range(20):
            assert np.allclose(stacked[i, 1], vol[i]), \
                f"Center channel mismatch at slice {i}"

    def test_neighbor_channels_correct(self):
        vol = np.arange(10 * 8 * 8, dtype=np.float32).reshape(10, 8, 8)
        stacked = stack_25d(vol, 3)
        # Slice 5: prev=vol[4], current=vol[5], next=vol[6]
        assert np.allclose(stacked[5, 0], vol[4])
        assert np.allclose(stacked[5, 1], vol[5])
        assert np.allclose(stacked[5, 2], vol[6])

    def test_even_stack_size_raises(self):
        with pytest.raises(AssertionError):
            stack_25d(np.ones((10, 16, 16), np.float32), stack_size=2)


# ============================================================================
# Test: 2.5D Augmentation Alignment
# ============================================================================

class TestAugmentationAlignment:
    def test_augment_on_stacked_channels_shape(self):
        """Augmenting (H, W, stack_size) must preserve correct output shape."""
        try:
            import albumentations as A
        except ImportError:
            pytest.skip("albumentations not installed")

        augment_fn = A.Compose([A.HorizontalFlip(p=1.0)])
        volume  = np.random.rand(12, 16, 16).astype(np.float32)
        stacked = stack_25d(volume, stack_size=3)  # (12, 3, 16, 16)

        result = np.stack([
            augment_fn(image=stacked[i].transpose(1, 2, 0))["image"].transpose(2, 0, 1)
            for i in range(12)
        ])
        assert result.shape == (12, 3, 16, 16)

    def test_neighbor_channel_relationship_before_augment(self):
        """stack_25d must produce correct channel-neighbor mapping."""
        vol = np.arange(10 * 8 * 8, dtype=np.float32).reshape(10, 8, 8)
        stacked = stack_25d(vol, 3)
        assert np.allclose(stacked[5, 0], vol[4])  # prev channel
        assert np.allclose(stacked[5, 1], vol[5])  # current channel
        assert np.allclose(stacked[5, 2], vol[6])  # next channel


# ============================================================================
# Test: synovitis_soft wiring
# ============================================================================

class TestSynovitisSoftTarget:
    SYN_IDX = LABEL_COLS.index("Synovitis")

    def _make_row(
        self, is_gold: int, synovitis: float, synovitis_soft: float
    ) -> pd.Series:
        data = {col: 0.0 for col in LABEL_COLS}
        data.update({
            "StudyInstanceUID": "TEST",
            "is_gold": is_gold,
            "Synovitis": synovitis,
            "synovitis_soft": synovitis_soft,
        })
        return pd.Series(data)

    def test_soft_target_for_nongold_row(self):
        """Non-gold rows must use synovitis_soft (0.22 or 0.63), not hard label."""
        from src.datasets.mri_dataset import _build_labels
        labels = _build_labels(self._make_row(0, 0, 0.63), LABEL_COLS)
        assert abs(labels[self.SYN_IDX].item() - 0.63) < 1e-4

    def test_hard_target_for_gold_row(self):
        """Gold rows must use the authoritative hard integer label."""
        from src.datasets.mri_dataset import _build_labels
        labels = _build_labels(self._make_row(1, 1.0, 0.22), LABEL_COLS)
        assert abs(labels[self.SYN_IDX].item() - 1.0) < 1e-4

    def test_other_labels_unaffected(self):
        """synovitis_soft routing must not affect other label indices."""
        from src.datasets.mri_dataset import _build_labels
        labels = _build_labels(self._make_row(0, 0, 0.63), LABEL_COLS)
        # All non-Synovitis labels were set to 0.0
        for i, col in enumerate(LABEL_COLS):
            if col != "Synovitis":
                assert abs(labels[i].item() - 0.0) < 1e-4, \
                    f"Label {col} at idx {i} unexpectedly modified"


# ============================================================================
# Test: Bilateral study handling -- keep-all strategy (v3)
# ============================================================================

class TestBilateralHandling:
    def test_bilateral_warning_emitted(self):
        """Studies with >12 series should emit a bilateral warning."""
        ds = KneeMRIDataset.__new__(KneeMRIDataset)
        mock_series = [MagicMock() for _ in range(15)]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds._check_bilateral(mock_series, "BILATERAL_STUDY")
            assert len(w) == 1
            assert "bilateral" in str(w[0].message).lower()

    def test_bilateral_does_not_filter_series(self):
        """_check_bilateral must NOT modify or filter the series list."""
        ds = KneeMRIDataset.__new__(KneeMRIDataset)
        mock_series = [MagicMock() for _ in range(15)]
        original_count = len(mock_series)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ds._check_bilateral(mock_series, "BILATERAL_STUDY")

        assert len(mock_series) == original_count, \
            f"Bilateral check modified series list! {original_count} -> {len(mock_series)}"

    def test_non_bilateral_no_warning(self):
        """Studies with <=12 series must not trigger the bilateral warning."""
        ds = KneeMRIDataset.__new__(KneeMRIDataset)
        mock_series = [MagicMock() for _ in range(8)]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds._check_bilateral(mock_series, "NORMAL_STUDY")
            assert len(w) == 0, f"Unexpected warning for normal study: {w}"


# ============================================================================
# Test: KneeMILModel
# ============================================================================

class TestKneeMILModel:
    def test_forward_output_shape(self, dummy_model, dummy_planes):
        logits, attn = dummy_model(dummy_planes)
        assert logits.shape == (2, 12)

    def test_raw_logits_no_sigmoid(
        self, dummy_model, dummy_planes
    ):
        """Output should contain values outside [0,1] (raw logits, not sigmoid)."""
        logits, _ = dummy_model(dummy_planes)
        # With random weights, some logits will be outside [0,1]
        assert ((logits > 1.0) | (logits < 0.0)).any().item(), \
            "Logits appear to be sigmoid-activated. Model must output raw logits."

    def test_attention_weights_sum_to_one(
        self, dummy_model, dummy_planes
    ):
        _, attn = dummy_model(dummy_planes)
        for plane_name, weights in attn.items():
            assert torch.allclose(weights.sum(-1), torch.ones(2), atol=1e-5), \
                f"Attention weights for {plane_name} don't sum to 1"

    def test_plane_mask_zeroes_missing_plane(
        self, dummy_model, dummy_planes
    ):
        """Masking out a plane should change the output logits."""
        mask_all = torch.ones(2, 3, dtype=torch.bool)
        mask_no_axial = mask_all.clone()
        mask_no_axial[:, 2] = False

        logits_full, _   = dummy_model(dummy_planes, plane_mask=mask_all)
        logits_masked, _ = dummy_model(dummy_planes, plane_mask=mask_no_axial)

        assert not torch.allclose(logits_full, logits_masked), \
            "Plane masking had no effect -- check zero-masking in forward()"

    def test_forward_without_plane_mask(self, dummy_model, dummy_planes):
        """Forward without plane_mask should not raise."""
        logits, _ = dummy_model(dummy_planes, plane_mask=None)
        assert logits.shape == (2, 12)

    def test_backbone_freeze_unfreezes(self, dummy_planes):
        """Backbone should be unfrozen after freeze_backbone_epochs."""
        model = KneeMILModel(
            backbone_name="efficientnet_b0",
            n_classes=12, stack_size=3, pretrained=False,
            use_grad_checkpointing=False,
            freeze_backbone_epochs=2,
        )
        # Check frozen initially
        for p in model.backbone.parameters():
            assert not p.requires_grad, "Backbone should be frozen initially"

        # Call on_epoch_start to trigger unfreeze
        model.on_epoch_start(2)
        for p in model.backbone.parameters():
            assert p.requires_grad, "Backbone should be unfrozen at epoch 2"


# ============================================================================
# Test: AttentionPooling
# ============================================================================

class TestAttentionPooling:
    def test_output_shape(self):
        pooler = AttentionPooling(embed_dim=32, hidden_dim=16)
        pooled, attn = pooler(torch.randn(4, 8, 32))
        assert pooled.shape == (4, 32)
        assert attn.shape  == (4, 8)

    def test_attention_softmax_constraint(self):
        pooler = AttentionPooling(embed_dim=32)
        _, attn = pooler(torch.randn(3, 6, 32))
        assert torch.allclose(attn.sum(-1), torch.ones(3), atol=1e-5), \
            "Attention weights must sum to 1 (softmax)"

    def test_single_instance_bag(self):
        """Single-instance bag (N=1) should work without errors."""
        pooler = AttentionPooling(embed_dim=16)
        pooled, attn = pooler(torch.randn(2, 1, 16))
        assert pooled.shape == (2, 16)
        assert torch.allclose(attn.sum(-1), torch.ones(2), atol=1e-5)


# ============================================================================
# Test: MaskedBCEWithLogitsLoss
# ============================================================================

class TestMaskedBCEWithLogitsLoss:
    def test_valid_labels_positive_finite_loss(self):
        crit = MaskedBCEWithLogitsLoss(gold_weight=1.0)
        loss = crit(
            torch.randn(4, 12),
            torch.randint(0, 2, (4, 12)).float(),
            torch.zeros(4, dtype=torch.long),
        )
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_all_nan_targets_zero_loss(self):
        """When all labels are NaN (masked), loss should be 0."""
        crit = MaskedBCEWithLogitsLoss(gold_weight=1.0)
        loss = crit(
            torch.randn(4, 12),
            torch.full((4, 12), float("nan")),
            torch.zeros(4, dtype=torch.long),
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_gold_weight_increases_loss(self):
        """Loss with gold_weight=10 on gold studies > loss with gold_weight=1."""
        torch.manual_seed(42)
        logits  = torch.randn(4, 12)
        targets = torch.randint(0, 2, (4, 12)).float()

        crit_high = MaskedBCEWithLogitsLoss(gold_weight=10.0)
        crit_low  = MaskedBCEWithLogitsLoss(gold_weight=1.0)

        loss_high = crit_high(
            logits, targets, torch.ones(4, dtype=torch.long)
        ).item()
        loss_low = crit_low(
            logits, targets, torch.zeros(4, dtype=torch.long)
        ).item()

        assert loss_high > loss_low

    def test_partial_nan_finite_loss(self):
        """Mixed valid+NaN labels should produce finite loss."""
        crit    = MaskedBCEWithLogitsLoss(gold_weight=1.0)
        targets = torch.ones(2, 12)
        targets[0, :6] = float("nan")  # First 6 labels masked for study 0
        loss = crit(
            torch.zeros(2, 12), targets, torch.zeros(2, dtype=torch.long)
        )
        assert torch.isfinite(loss)

    def test_no_nan_propagation(self):
        """NaN in targets must not propagate to the loss gradient."""
        crit    = MaskedBCEWithLogitsLoss(gold_weight=1.0)
        logits  = torch.randn(3, 12, requires_grad=True)
        targets = torch.randint(0, 2, (3, 12)).float()
        targets[:, :3] = float("nan")

        loss = crit(logits, targets, torch.zeros(3, dtype=torch.long))
        loss.backward()
        assert not torch.isnan(logits.grad).any(), "NaN propagated to gradients"


# ============================================================================
# Test: compute_per_label_auc
# ============================================================================

class TestComputePerLabelAUC:
    LABELS = [f"label_{i}" for i in range(12)]

    def test_perfect_predictions_auc_one(self):
        targets = np.zeros((100, 12))
        targets[:50] = 1.0
        logits = np.where(targets == 1.0, 5.0, -5.0)
        scores = compute_per_label_auc(logits, targets, self.LABELS)
        assert scores["macro_auc"] == pytest.approx(1.0, abs=1e-4)

    def test_nan_labels_excluded_from_auc(self):
        targets = np.full((100, 12), float("nan"))
        targets[:, 0] = np.where(np.arange(100) < 50, 1.0, 0.0)
        logits = np.nan_to_num(
            np.where(targets == 1.0, 5.0, -5.0), nan=-5.0
        )
        scores = compute_per_label_auc(logits, targets, self.LABELS)
        assert not np.isnan(scores["label_0"])
        for i in range(1, 12):
            assert np.isnan(scores[f"label_{i}"]), \
                f"label_{i} should be NaN (fully masked)"

    def test_macro_averages_only_valid_labels(self):
        targets = np.zeros((100, 12))
        targets[:50, 0] = 1.0  # Only label_0 has valid targets
        logits = np.where(targets > 0, 3.0, -3.0)
        scores = compute_per_label_auc(logits, targets, self.LABELS)
        # macro_auc should equal label_0 since it's the only valid label
        assert scores["macro_auc"] == pytest.approx(scores["label_0"], abs=1e-4)

    def test_all_same_class_returns_nan(self):
        """Single-class labels (all positive or all negative) should return NaN AUC."""
        targets = np.ones((100, 12))  # All positive -- AUC undefined
        logits  = np.random.randn(100, 12)
        scores  = compute_per_label_auc(logits, targets, self.LABELS)
        # All labels have only one class -> all NaN -> macro_auc NaN
        assert np.isnan(scores["macro_auc"])
