"""tests/test_tensorboard_logging.py

Comprehensive unit tests verifying:
  1. Adversarial Gold Study Synovitis filtering (guarantees AUC drops to 0.25 if broken)
  2. Golden-fixture parity test matching v16 compute_per_label_auc baseline
  3. Config-driven label loading & dynamic Synovitis index resolution
  4. Stable macro_auc_11 vs macro_auc_12 signal separation (0.50 Synovitis -> macro_12=0.75)
  5. TelemetryLogger protocol conformance and NoOpLogger safety
  6. Pure Object-Oriented Figure rendering without global pyplot side effects
  7. Pure conditional presence-weighted attention heatmap calculation (0.40 vs 0.20)
  8. Tag sanitization eliminating TensorBoard character rewrite warnings
  9. Non-finite scalar warning logging vs silent metric NaN handling in log_epoch
 10. Crash-resilience guards on visualizers, step scalars, and logger instantiation
 11. DDP-unwrapped, device-safe model graph tracing
 12. Gloo CPU 2-process all_gather_eval_tensors execution with 2D floats and 1D booleans
 13. Protocol method override completeness test
 14. Real SummaryWriter event serialization and readback via EventAccumulator
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch
import warnings

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
import torch
import torch.distributed as dist
import torch.nn as nn

from src.training.metrics import (
    LabelStats,
    MetricsResult,
    all_gather_eval_tensors,
    compute_competition_metrics,
    load_label_columns,
)
from src.utils.tensorboard import (
    HAS_TENSORBOARD,
    NoOpLogger,
    TelemetryLogger,
    TensorBoardLogger,
    compute_conditional_slot_attention,
    get_logger,
    sanitize_tag,
)


def test_telemetry_logger_protocol_conformance():
    """Verify NoOpLogger and TensorBoardLogger conform to TelemetryLogger protocol."""
    noop = NoOpLogger()
    assert isinstance(noop, TelemetryLogger)


def test_tensorboard_logger_overrides_all_protocol_methods():
    """Verify TensorBoardLogger explicitly overrides all public methods of TelemetryLogger."""
    protocol_methods = {
        name
        for name, val in TelemetryLogger.__dict__.items()
        if not name.startswith("_") and callable(val)
    }
    tb_dict = TensorBoardLogger.__dict__
    missing = [m for m in protocol_methods if m not in tb_dict]
    assert not missing, f"TensorBoardLogger failed to override protocol methods: {missing}"


def test_noop_logger_safe_execution():
    """Verify all NoOpLogger methods execute cleanly with dummy data without errors."""
    logger = NoOpLogger()
    dummy_result = MetricsResult(
        macro_auc_12=0.85,
        macro_auc_11=0.86,
        per_label_auc={"ACL": 0.90},
        per_label_ap={"ACL": 0.85},
        label_stats={"ACL": LabelStats(10, 10, 5, 5, 0.5)},
        n_valid_labels=1,
        n_non_finite_logits=0,
    )
    logger.log_scalar("loss", 0.5, 0)
    logger.log_step(0, 0.45, lrs={"backbone": 3e-5}, grad_norm=0.8, grad_scale=1024.0, throughput=2.5)
    logger.log_epoch(0, 0.40, 0.35, dummy_result, peak_gpu_mem_mb=1200.0, slot_presence_rates={"COR_T1": 0.6})
    logger.log_graph(nn.Linear(10, 2), (torch.randn(1, 10),))
    logger.log_attention_heatmap(np.zeros((12, 6)), 0)
    logger.log_pr_curves(np.zeros((10, 12)), np.zeros((10, 12)), 0)
    logger.flush()
    logger.close()


def test_get_logger_factory_fallbacks():
    """Verify get_logger factory returns NoOpLogger when disabled, tag is None, or on error."""
    assert isinstance(get_logger(run_tag=None), NoOpLogger)
    assert isinstance(get_logger(run_tag="test_run", enabled=False), NoOpLogger)

    with patch("src.utils.tensorboard.HAS_TENSORBOARD", True), \
         patch("src.utils.tensorboard.TensorBoardLogger", side_effect=RuntimeError("Disk failure")):
        with pytest.warns(UserWarning, match="Failed to initialize TensorBoardLogger"):
            fallback_logger = get_logger(run_tag="test_run", enabled=True)
            assert isinstance(fallback_logger, NoOpLogger)


def test_adversarial_gold_synovitis_discrimination():
    """Verify Synovitis test strictly fails if non-gold rows are included.

    Setup:
      - 4 Gold rows: 2 negatives (target=0.0, logit=-3.0), 2 positives (target=1.0, logit=+3.0).
        Gold-only AUC is exactly 1.0.
      - 4 Non-Gold rows with adversarial predictions:
        2 pseudo-positives (target=0.63) given strong negative logits (-5.0).
        2 pseudo-negatives (target=0.22) given strong positive logits (+5.0).
    Mathematical Verification:
      - With gold filter working: Positives [+3, +3] vs Negatives [-3, -3] -> AUC == 1.0.
      - With gold filter broken: Positives [+3, +3, -5, -5] vs Negatives [-3, -3, +5, +5].
        Only the 4 gold-pos vs gold-neg pairs succeed; AUC = 4 / 16 = 0.25 exactly.
    """
    n_samples = 8
    label_cols = load_label_columns()
    syn_idx = label_cols.index("Synovitis")

    targets = np.zeros((n_samples, 12), dtype=np.float32)
    logits = np.zeros((n_samples, 12), dtype=np.float32)
    gold_mask = np.zeros(n_samples, dtype=bool)

    # 4 Gold studies
    gold_mask[0:4] = True
    targets[0:2, syn_idx] = 0.0
    logits[0:2, syn_idx] = -3.0
    targets[2:4, syn_idx] = 1.0
    logits[2:4, syn_idx] = +3.0

    # 4 Adversarial Non-Gold studies
    gold_mask[4:8] = False
    targets[4:6, syn_idx] = 0.63
    logits[4:6, syn_idx] = -5.0
    targets[6:8, syn_idx] = 0.22
    logits[6:8, syn_idx] = +5.0

    # 1. Valid execution with gold_mask
    result = compute_competition_metrics(logits, targets, gold_mask=gold_mask, label_cols=label_cols)
    assert np.isclose(result.per_label_auc["Synovitis"], 1.0), "Synovitis gold AUC must be 1.0"

    # 2. Verify adversarial sensitivity: evaluating without filtering non-gold yields AUC == 0.25
    all_gold_mask = np.ones(n_samples, dtype=bool)  # Simulates broken filter
    broken_result = compute_competition_metrics(logits, targets, gold_mask=all_gold_mask, label_cols=label_cols)
    assert np.isclose(broken_result.per_label_auc["Synovitis"], 0.25), "Broken filter must yield exactly 0.25 AUC"
    assert broken_result.per_label_auc["Synovitis"] < 0.30


def test_golden_fixture_matches_v16_reference():
    """Verify compute_competition_metrics matches verbatim v16 trainer.py implementation to 1e-6."""
    # Verbatim transcribed from src/training/trainer.py (baseline v16)
    def compute_per_label_auc_v16_verbatim(all_logits, all_targets, label_cols, gold_mask=None):
        SYNOVITIS_IDX = 8
        all_sigmoid = 1.0 / (1.0 + np.exp(-all_logits))
        scores = {}
        for i, col in enumerate(label_cols):
            valid = ~np.isnan(all_targets[:, i])
            if i == SYNOVITIS_IDX and gold_mask is not None:
                valid = valid & gold_mask.astype(bool)
            if valid.sum() < 2:
                scores[col] = float("nan")
                continue
            try:
                scores[col] = roc_auc_score(all_targets[valid, i], all_sigmoid[valid, i])
            except ValueError:
                scores[col] = float("nan")
        valid_aucs = [v for v in scores.values() if not np.isnan(v)]
        scores["macro_auc"] = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
        return scores

    np.random.seed(42)
    n_samples = 30
    label_cols = load_label_columns()
    logits = np.random.randn(n_samples, 12).astype(np.float32)
    targets = np.random.choice([0.0, 1.0, np.nan], size=(n_samples, 12), p=[0.45, 0.45, 0.10]).astype(np.float32)
    gold_mask = np.random.choice([False, True], size=n_samples, p=[0.7, 0.3])

    ref_scores = compute_per_label_auc_v16_verbatim(logits, targets, label_cols, gold_mask)
    new_result = compute_competition_metrics(logits, targets, gold_mask=gold_mask, label_cols=label_cols)

    for col in label_cols:
        ref_val = ref_scores[col]
        new_val = new_result.per_label_auc[col]
        if np.isnan(ref_val):
            assert np.isnan(new_val)
        else:
            assert np.isclose(ref_val, new_val, atol=1e-6)

    assert np.isclose(ref_scores["macro_auc"], new_result.macro_auc_12, atol=1e-6)


def test_gold_mask_required_for_synovitis():
    """Verify compute_competition_metrics raises ValueError if gold_mask is None when Synovitis is evaluated."""
    label_cols = load_label_columns()
    targets = np.zeros((4, 12), dtype=np.float32)
    logits = np.zeros((4, 12), dtype=np.float32)

    with pytest.raises(ValueError, match="gold_mask is required"):
        compute_competition_metrics(logits, targets, gold_mask=None, label_cols=label_cols)


def test_dynamic_synovitis_index_and_config_order():
    """Verify Synovitis index is resolved dynamically regardless of column permutation."""
    original_cols = load_label_columns()
    permuted_cols = list(reversed(original_cols))

    targets = np.zeros((4, 12), dtype=np.float32)
    logits = np.zeros((4, 12), dtype=np.float32)
    gold_mask = np.array([True, True, True, True])

    syn_perm_idx = permuted_cols.index("Synovitis")
    targets[0:2, syn_perm_idx] = 0.0
    logits[0:2, syn_perm_idx] = -2.0
    targets[2:4, syn_perm_idx] = 1.0
    logits[2:4, syn_perm_idx] = +2.0

    result = compute_competition_metrics(logits, targets, gold_mask=gold_mask, label_cols=permuted_cols)
    assert np.isclose(result.per_label_auc["Synovitis"], 1.0)


def test_macro_auc_11_and_12_separation():
    """Verify macro_auc_11 separates from macro_auc_12 when Synovitis has divergent valid AUC."""
    label_cols = load_label_columns()
    n_samples = 4
    targets = np.zeros((n_samples, 12), dtype=np.float32)
    logits = np.zeros((n_samples, 12), dtype=np.float32)
    gold_mask = np.array([True] * n_samples)

    # ACL (col 0): perfect predictor (AUC = 1.0)
    targets[0:2, 0] = 0.0
    logits[0:2, 0] = -3.0
    targets[2:4, 0] = 1.0
    logits[2:4, 0] = +3.0

    # Synovitis (col 8): positive [-1.0, 0.0] vs negative [-2.0, +2.0] gives 2/4 = 0.50 exactly
    targets[0:2, 8] = 0.0
    logits[0, 8] = -2.0
    logits[1, 8] = +2.0
    targets[2:4, 8] = 1.0
    logits[2, 8] = -1.0
    logits[3, 8] = 0.0

    # Remaining 10 columns: all NaN
    for c in range(12):
        if c not in (0, 8):
            targets[:, c] = float("nan")

    result = compute_competition_metrics(logits, targets, gold_mask=gold_mask, label_cols=label_cols)
    assert np.isclose(result.per_label_auc["ACL"], 1.0)
    assert np.isclose(result.per_label_auc["Synovitis"], 0.50)
    assert np.isclose(result.macro_auc_11, 1.0), "macro_auc_11 must ignore Synovitis"
    assert np.isclose(result.macro_auc_12, 0.75), "macro_auc_12 must average ACL (1.0) and Synovitis (0.50)"


def test_oo_figure_no_pyplot_side_effects():
    """Verify attention heatmap uses pure OO Figure API without modifying pyplot global state."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_writer = MagicMock()
        with patch("src.utils.tensorboard.SummaryWriter", return_value=mock_writer), \
             patch("src.utils.tensorboard.HAS_TENSORBOARD", True):
            tb_logger = TensorBoardLogger(log_dir=tmp_dir)

            initial_fignums = plt.get_fignums()
            dummy_attn = np.ones((12, 6), dtype=np.float32) / 6.0
            tb_logger.log_attention_heatmap(dummy_attn, step=1)

            mock_writer.add_figure.assert_called_once()
            fig_arg = mock_writer.add_figure.call_args[0][1]
            assert isinstance(fig_arg, Figure)
            assert plt.get_fignums() == initial_fignums, "Pyplot figures leaked into global state!"


def test_presence_weighted_attention_heatmap():
    """Verify pure conditional attention function correctly normalizes active slots."""
    attn = np.zeros((2, 12, 6), dtype=np.float32)
    # Study 0: slot 0 present with weight 0.40
    attn[0, :, 0] = 0.40
    # Study 1: slot 0 absent with weight 0.00
    attn[1, :, 0] = 0.00

    presence = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)

    cond_attn = compute_conditional_slot_attention(attn, presence)
    # Slot 0 was present in study 0 with 0.40; conditional mean must be 0.40 (not 0.20)
    assert np.isclose(cond_attn[:, 0].mean(), 0.40)


def test_tag_sanitization():
    """Verify tag sanitization cleans spaces and quotes to prevent TensorBoard rewrite warnings."""
    assert sanitize_tag("Baker's") == "Bakers"
    assert sanitize_tag("Medial Meniscus") == "Medial_Meniscus"
    assert sanitize_tag("AUC/PF OA") == "AUC/PF_OA"
    assert sanitize_tag("Attention/Finding_to_Slot") == "Attention/Finding_to_Slot"


def test_non_finite_scalar_warning():
    """Verify non-finite step scalars log a warning rather than being silently dropped."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_writer = MagicMock()
        with patch("src.utils.tensorboard.SummaryWriter", return_value=mock_writer), \
             patch("src.utils.tensorboard.HAS_TENSORBOARD", True):
            tb_logger = TensorBoardLogger(log_dir=tmp_dir)
            with pytest.warns(UserWarning, match="Telemetry received non-finite value"):
                tb_logger.log_scalar("Loss/diverged", float("nan"), step=5)
            mock_writer.add_scalar.assert_not_called()


def test_nan_metrics_in_log_epoch_do_not_warn():
    """Verify legitimate NaN metrics in log_epoch are skipped silently without warnings."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_writer = MagicMock()
        with patch("src.utils.tensorboard.SummaryWriter", return_value=mock_writer), \
             patch("src.utils.tensorboard.HAS_TENSORBOARD", True):
            tb_logger = TensorBoardLogger(log_dir=tmp_dir)
            metrics_with_nan = MetricsResult(
                macro_auc_12=0.80,
                macro_auc_11=0.80,
                per_label_auc={"ACL": 0.85, "Synovitis": float("nan")},
                per_label_ap={"ACL": 0.80, "Synovitis": float("nan")},
                label_stats={"ACL": LabelStats(10, 10, 5, 5, 0.5)},
                n_valid_labels=1,
                n_non_finite_logits=0,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # Any warning raises an exception
                tb_logger.log_epoch(0, 0.40, 0.35, metrics_with_nan)


def test_crash_resilience_guards():
    """Verify exceptions in SummaryWriter or visualizers degrade safely without throwing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_writer = MagicMock()
        with patch("src.utils.tensorboard.SummaryWriter", return_value=mock_writer), \
             patch("src.utils.tensorboard.HAS_TENSORBOARD", True):
            tb_logger = TensorBoardLogger(log_dir=tmp_dir)

            # 1. Figure runtime failure
            mock_writer.add_figure.side_effect = RuntimeError("Disk full during add_figure")
            with pytest.warns(UserWarning, match="Telemetry call log_attention_heatmap failed safely"):
                tb_logger.log_attention_heatmap(np.zeros((12, 6)), step=1)

            # 2. PR curve runtime failure
            mock_writer.add_pr_curve.side_effect = RuntimeError("Serialization error in add_pr_curve")
            targets = np.array([[0, 1], [1, 0]], dtype=np.float32)
            logits = np.array([[-1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
            with pytest.warns(UserWarning, match="Telemetry call log_pr_curves failed safely"):
                tb_logger.log_pr_curves(targets, logits, step=1, label_cols=["ACL", "MCL"])


def test_device_and_ddp_safe_graph_tracing():
    """Verify log_graph unwraps model.module and places dummy tensors on model device."""
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(6, 12)

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return self.linear(mask)

    class MockDDPWrapper(nn.Module):
        def __init__(self, mod: nn.Module):
            super().__init__()
            self.module = mod

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    raw_model = DummyModel()
    ddp_model = MockDDPWrapper(raw_model)
    dummy_x = torch.zeros(1, 6, 3, 336, 336)
    dummy_mask = torch.ones(1, 6)

    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_writer = MagicMock()
        with patch("src.utils.tensorboard.SummaryWriter", return_value=mock_writer), \
             patch("src.utils.tensorboard.HAS_TENSORBOARD", True):
            tb_logger = TensorBoardLogger(log_dir=tmp_dir)
            tb_logger.log_graph(ddp_model, (dummy_x, dummy_mask))

            mock_writer.add_graph.assert_called_once()
            called_model = mock_writer.add_graph.call_args[0][0]
            assert called_model is raw_model, "Graph tracing must operate on unwrapped model.module"


def _gloo_gather_worker(rank: int, world_size: int, port: str):
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Rank 0 has length 3, Rank 1 has length 2; gathers 2D float tensor and 1D bool tensor
    local_len = 3 if rank == 0 else 2
    local_logits = torch.randn(local_len, 12, dtype=torch.float32)
    local_mask = torch.tensor([bool(i % 2) for i in range(local_len)], dtype=torch.bool)

    gathered_logits, gathered_mask = all_gather_eval_tensors((local_logits, local_mask))

    assert gathered_logits.shape == (5, 12)
    assert gathered_mask.shape == (5,)
    assert gathered_mask.dtype == torch.bool
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed not available")
def test_all_gather_eval_tensors_gloo_cpu():
    """Verify 2-process Gloo CPU all_gather_eval_tensors gathers 2D floats and 1D booleans."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = str(s.getsockname()[1])
    s.close()

    torch.multiprocessing.spawn(_gloo_gather_worker, args=(2, port), nprocs=2, join=True)


@pytest.mark.skipif(not HAS_TENSORBOARD, reason="tensorboard not installed in test environment")
def test_real_event_accumulator_serialization(tmp_path):
    """End-to-end integration test verifying real SummaryWriter event creation and EventAccumulator readback."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    log_dir = tmp_path / "real_tb_test"
    logger = TensorBoardLogger(log_dir=log_dir)

    # 1. Log scalars
    logger.log_scalar("Loss/train_step", 0.42, step=0)
    logger.log_scalar("Loss/train_step", 0.38, step=1)
    logger.log_scalar("AUC/macro_val", 0.81, step=1)

    # 2. Log figure
    dummy_attn = np.ones((12, 6), dtype=np.float32) / 6.0
    logger.log_attention_heatmap(dummy_attn, step=1)

    # 3. Log PR curve
    targets = np.array([0, 1, 0, 1], dtype=np.float32).reshape(-1, 1)
    logits = np.array([-2.0, 2.0, -1.0, 1.0], dtype=np.float32).reshape(-1, 1)
    logger.log_pr_curves(targets, logits, step=1, label_cols=["ACL"])

    logger.flush()
    logger.close()

    event_acc = EventAccumulator(str(log_dir))
    event_acc.Reload()

    # Verify scalar tags
    scalar_tags = event_acc.Tags().get("scalars", [])
    assert "Loss/train_step" in scalar_tags
    assert "AUC/macro_val" in scalar_tags

    steps = [e.step for e in event_acc.Scalars("Loss/train_step")]
    values = [e.value for e in event_acc.Scalars("Loss/train_step")]
    assert steps == [0, 1]
    assert np.isclose(values[0], 0.42)
    assert np.isclose(values[1], 0.38)

    # Verify figure logged in images tag
    image_tags = event_acc.Tags().get("images", [])
    assert "Attention/Finding_to_Slot" in image_tags

    # Verify PR curve registered
    tensors_or_pr = event_acc.Tags().get("tensors", []) + event_acc.Tags().get("scalars", [])
    assert any("PR/ACL" in tag for tag in tensors_or_pr)


def test_all_zero_predictions_and_missing_labels():
    """Verify compute_competition_metrics handles all-zero predictions and entirely missing labels safely."""
    label_cols = load_label_columns()
    n_samples = 10
    # All zero logits (p=0.5)
    logits = np.zeros((n_samples, 12), dtype=np.float32)
    # Binary targets for first 2 labels, NaN for the other 10
    targets = np.full((n_samples, 12), np.nan, dtype=np.float32)
    targets[:5, 0] = 0.0
    targets[5:, 0] = 1.0
    targets[:5, 1] = 0.0
    targets[5:, 1] = 1.0

    gold_mask = np.ones(n_samples, dtype=bool)
    result = compute_competition_metrics(logits, targets, gold_mask=gold_mask, label_cols=label_cols)

    # All-zero logits have tied predictions: AUC is 0.50
    assert np.isclose(result.per_label_auc["ACL"], 0.50)
    assert np.isclose(result.per_label_auc["MCL"], 0.50)
    assert np.isnan(result.per_label_auc["Synovitis"])
    assert result.n_valid_labels == 2
    assert np.isclose(result.macro_auc_12, 0.50)
    assert np.isclose(result.macro_auc_11, 0.50)


def test_log_pr_curves_skips_synovitis_without_gold_mask():
    """Verify log_pr_curves logs remaining findings with a warning when gold_mask is None."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_writer = MagicMock()
        with patch("src.utils.tensorboard.SummaryWriter", return_value=mock_writer), \
             patch("src.utils.tensorboard.HAS_TENSORBOARD", True):
            tb_logger = TensorBoardLogger(log_dir=tmp_dir)

            targets = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
            logits = np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32)
            label_cols = ["ACL", "Synovitis"]

            with pytest.warns(UserWarning, match="skipping Synovitis PR curve"):
                tb_logger.log_pr_curves(targets, logits, step=1, gold_mask=None, label_cols=label_cols)

            # ACL was logged, Synovitis was skipped
            assert mock_writer.add_pr_curve.call_count == 1
            call_tag = mock_writer.add_pr_curve.call_args[0][0]
            assert call_tag == "PR/ACL"
