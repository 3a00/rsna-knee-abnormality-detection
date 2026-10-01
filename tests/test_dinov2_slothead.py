"""tests/test_dinov2_slothead.py

Refined unit tests verifying DINOv2-Small SlotHead architecture:
  - Input/output shape contracts (B, 12 raw logits in FP32)
  - Native 336x336 resolution processing
  - Present-slot gathering efficiency (absent slots never touch the backbone)
  - Unconditional FP32 SlotHead stability under mixed precision
  - CPU mixed-precision smoke test
  - CUDA FP16 vs FP32 logit parity and active GradScaler verification (CUDA-gated)
  - State dict round-trip inference parity for offline submission environments
  - Offline weights loading with bicubic pos_embed resampling (518px -> 112px)
  - DDP all-absent batch gradient synchronization guard
  - Diagnostic forward_with_attention contract, parity, and absent-slot zeroing
  - Statistical slot dropout verification with safety invariants (seeded)
  - Bias initialization from 12 distinct label priors with boundary clamping and hardcoded check
  - Absent-slot pixel safety (NaN/Inf immunity)
  - Gradient flow across all trainable parameter groups
"""

from __future__ import annotations

import io
import math
import os
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import timm
import torch

from src.models.dinov2_slothead import DINOv2SlotHead

# Set global seed for reproducible test batch synthesis
torch.manual_seed(42)


def test_model_forward_shapes():
    """Verify input returns unscaled raw logits of shape (B, 12) as a single Tensor."""
    model = DINOv2SlotHead(img_size=112, pretrained=False)
    model.eval()

    b, s = 2, 6
    x = torch.randn(b, s, 3, 112, 112)
    mask = torch.ones(b, s)

    with torch.no_grad():
        out = model(x, mask)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (b, 12)
    assert torch.isfinite(out).all()


def test_model_336_resolution():
    """Verify model processes native 336x336 resolution input."""
    model = DINOv2SlotHead(img_size=336, pretrained=False)
    model.eval()

    x = torch.randn(1, 6, 3, 336, 336)
    mask = torch.ones(1, 6)

    with torch.no_grad():
        out = model(x, mask)

    assert out.shape == (1, 12)
    assert torch.isfinite(out).all()


def test_present_slot_gathering_efficiency():
    """Verify absent slots are never forwarded through the backbone."""
    model = DINOv2SlotHead(img_size=112, pretrained=False)
    model.eval()

    calls = []
    orig_forward = model.backbone.forward

    def hook_forward(x_in):
        calls.append(x_in.shape)
        return orig_forward(x_in)

    model.backbone.forward = hook_forward

    x = torch.randn(2, 6, 3, 112, 112)
    mask = torch.tensor([
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    ])

    with torch.no_grad():
        logits = model(x, mask)

    assert len(calls) == 1
    # Backbone must be called with exactly 5 slices, NOT 2 * 6 = 12 slices!
    assert calls[0] == torch.Size([5, 3, 112, 112])
    assert logits.shape == (2, 12)


def test_unconditional_fp32_head_stability():
    """Verify SlotHead outputs FP32 and remains stable even under mixed precision."""
    model = DINOv2SlotHead(img_size=112, pretrained=False)
    model.eval()

    x = torch.randn(2, 6, 3, 112, 112)
    mask = torch.ones(2, 6)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(x, mask)

    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_cpu_amp_smoke():
    """Verify CPU mixed-precision forward and backward pass smoke test."""
    model = DINOv2SlotHead(img_size=112, pretrained=False)
    model.train()

    x = torch.randn(2, 6, 3, 112, 112)
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])

    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(x, mask)
        loss = logits.sum()

    loss.backward()
    assert torch.isfinite(logits).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU for real FP16 and GradScaler validation")
def test_cuda_fp16_fp32_parity_and_gradscaler():
    """Verify logit parity across FP16 and FP32 at 336px with calibrated GradScaler on CUDA."""
    device = torch.device("cuda")
    model = DINOv2SlotHead(img_size=336, pretrained=False).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(2, 6, 3, 336, 336, device=device)
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]], device=device)

    # FP32 forward
    with torch.no_grad():
        logits_fp32 = model(x, mask)

    # CUDA FP16 forward
    with torch.autocast("cuda", dtype=torch.float16):
        logits_fp16 = model(x, mask)
        loss = logits_fp16.sum()

    # Logit parity check on CUDA
    max_delta = (logits_fp32 - logits_fp16).abs().max().item()
    assert max_delta < 5e-2, f"CUDA FP16 vs FP32 logit delta too large: {max_delta}"

    # Calibrated GradScaler check (init_scale=1024.0 prevents false overflow assertions)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Unscaled gradient for {name} is non-finite!"
    scaler.step(optimizer)
    scaler.update()


def test_state_dict_roundtrip_inference_parity():
    """Verify state_dict save/load round-trip produces exact identical logits for offline inference."""
    model_a = DINOv2SlotHead(img_size=112, pretrained=False)
    model_a.eval()

    buffer = io.BytesIO()
    torch.save(model_a.state_dict(), buffer)
    buffer.seek(0)

    model_b = DINOv2SlotHead(img_size=112, pretrained=False)
    model_b.eval()
    model_b.load_state_dict(torch.load(buffer, weights_only=True), strict=True)

    x = torch.randn(2, 6, 3, 112, 112)
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]])

    with torch.no_grad():
        logits_a = model_a(x, mask)
        logits_b = model_b(x, mask)

    assert torch.equal(logits_a, logits_b), "State dict round-trip produced mismatched logits!"


def test_offline_weights_resampling_clean_load():
    """Verify offline weights load cleanly with timm bicubic pos_embed resampling (518px -> 112px)."""
    # Create temporary 518px checkpoint file
    m518 = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0, img_size=518)
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        tmp_path = f.name
        torch.save(m518.state_dict(), tmp_path)

    try:
        # Load into DINOv2SlotHead at 112px resolution using weights_path
        model_resampled = DINOv2SlotHead(img_size=112, pretrained=False, weights_path=tmp_path)
        model_resampled.eval()

        # Target 112px pos_embed should have shape [1, 1 + (112/14)^2, 384] = [1, 65, 384]
        assert model_resampled.backbone.pos_embed.shape == (1, 65, 384)

        # Forward pass runs cleanly
        x = torch.randn(1, 6, 3, 112, 112)
        mask = torch.ones(1, 6)
        with torch.no_grad():
            out = model_resampled(x, mask)
        assert out.shape == (1, 12)
        assert torch.isfinite(out).all()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_all_absent_batch_ddp_guard():
    """Verify an all-absent batch executes dummy forward and preserves gradient graph."""
    model = DINOv2SlotHead(img_size=112, pretrained=False)
    model.train()

    x = torch.randn(2, 6, 3, 112, 112)
    all_absent = torch.zeros(2, 6)

    logits = model(x, all_absent)
    assert logits.shape == (2, 12)
    assert torch.isfinite(logits).all()

    loss = logits.sum()
    loss.backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} gradient is None on all-absent batch!"
            assert torch.isfinite(p.grad).all()


def test_forward_with_attention_contract():
    """Verify forward_with_attention contract, parity with forward, and absent zeroing."""
    model = DINOv2SlotHead(img_size=112, pretrained=False)
    model.eval()

    b, s = 2, 6
    x = torch.randn(b, s, 3, 112, 112)
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]])

    with torch.no_grad():
        logits_std = model(x, mask)
        logits_diag, attn = model.forward_with_attention(x, mask)

    assert torch.allclose(logits_std, logits_diag, atol=1e-5), "Parity mismatch between forward methods!"
    assert attn.shape == (b, 12, s)
    assert torch.isfinite(attn).all()

    # Absent slots strictly 0.0
    assert (attn[0, :, 5] == 0.0).all()
    assert (attn[1, :, 1] == 0.0).all()

    # Present slots sum to 1.0 per finding
    assert torch.allclose(attn[0].sum(dim=-1), torch.ones(12), atol=1e-5)
    assert torch.allclose(attn[1].sum(dim=-1), torch.ones(12), atol=1e-5)

    # All-absent study attention rows strictly 0.0
    all_absent_mask = torch.zeros(1, 6)
    with torch.no_grad():
        _, attn_all_absent = model.forward_with_attention(x[:1], all_absent_mask)
    assert (attn_all_absent == 0.0).all(), "All-absent study should have zero attention weights across all slots!"


def test_slot_dropout_statistical_and_safety_invariants():
    """Verify slot dropout statistically and assert safety invariants including deterministic edge case."""
    torch.manual_seed(42)
    p_drop = 0.20
    model = DINOv2SlotHead(img_size=112, pretrained=False, slot_dropout_p=p_drop)

    # In eval mode, no dropout occurs
    model.eval()
    present_fixed = torch.tensor([[True, True, True, False, False, False]])
    assert (model._apply_slot_dropout(present_fixed) == present_fixed).all()

    # In train mode, test over 500 trials
    model.train()
    trials = 500
    present_batch = torch.tensor([[True, True, True, True, False, False]]).repeat(trials, 1)

    dropped_batch = model._apply_slot_dropout(present_batch)

    # Invariant 1: originally absent slots (indices 4, 5) NEVER become present
    assert (dropped_batch[:, 4:] == False).all(), "Absent slots became present!"

    # Invariant 2: at least 1 slot remains present per row
    assert dropped_batch[:, :4].any(dim=-1).all(), "All slots dropped in a row!"

    # Invariant 3: empirical drop rate on present slots is close to p_drop
    active_drops = (~dropped_batch[:, :4] & present_batch[:, :4]).float().mean().item()
    assert 0.14 <= active_drops <= 0.26, f"Empirical drop rate {active_drops} outside expected margin"

    # Invariant 4: completely absent row stays completely absent
    all_absent = torch.zeros(10, 6, dtype=torch.bool)
    assert (model._apply_slot_dropout(all_absent) == False).all()

    # Invariant 5 (deterministic edge case): single present slot at p=1.0 is preserved by the restoration branch
    model_full_drop = DINOv2SlotHead(img_size=112, pretrained=False, slot_dropout_p=1.0)
    model_full_drop.train()
    single_present = torch.tensor([[True, False, False, False, False, False]])
    restored_single = model_full_drop._apply_slot_dropout(single_present)
    assert (restored_single == single_present).all(), "Single present slot failed to restore under p=1.0!"


def test_bias_prior_initialization():
    """Verify cls_b initialized from 12 distinct priors matches log-odds with literal check and clamping."""
    # 12 distinct realistic knee finding prevalence rates + boundary cases
    priors = [0.18, 0.02, 0.40, 0.25, 0.30, 0.15, 0.20, 0.35, 0.10, 0.05, 0.0, 1.0]
    model = DINOv2SlotHead(img_size=112, pretrained=False, init_bias_priors=priors)

    # Hardcoded literal check: p=0.18 -> log(0.18 / 0.82) = -1.516347...
    expected_p0 = -1.516347
    assert torch.isclose(model.cls_b[0], torch.tensor(expected_p0, dtype=torch.float32), atol=1e-4)

    clamped = torch.as_tensor(priors, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
    expected_bias = torch.log(clamped / (1.0 - clamped))
    assert torch.allclose(model.cls_b, expected_bias, atol=1e-5)

    # Check boundary clamping values
    assert model.cls_b[10].item() < -9.0  # Clamped near 1e-4
    assert model.cls_b[11].item() > 9.0   # Clamped near 1 - 1e-4


def test_absent_slot_pixel_safety():
    """Verify NaN/Inf pixels in absent slots never reach the backbone or gradients."""
    model = DINOv2SlotHead(img_size=112, pretrained=False)
    model.train()

    x = torch.randn(2, 6, 3, 112, 112)
    x[0, 5] = float("nan")  # Slot 5 is absent
    x[1, 2] = float("inf")  # Slot 2 is absent
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]])

    logits = model(x, mask)
    assert torch.isfinite(logits).all()

    loss = logits.sum()
    loss.backward()

    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Gradient for {name} poisoned by absent-slot NaN pixel!"


def test_gradient_flow_trainable_parameters():
    """Verify gradients propagate to all trainable parameters."""
    model = DINOv2SlotHead(img_size=112, pretrained=False)
    model.train()

    x = torch.randn(2, 6, 3, 112, 112)
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]])

    logits = model(x, mask)
    loss = logits.sum()
    loss.backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Gradient is None for {name}"
            assert torch.isfinite(p.grad).all(), f"Gradient non-finite for {name}"
            assert (p.grad != 0).any(), f"Gradient all zeros for {name}"
