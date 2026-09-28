#!/usr/bin/env python3
"""
scripts/smoke_efficiency_tracer.py

End-to-End Wiring Smoke Tracer for RSNA Knee Abnormality Detection Efficiency Track (Final Round).
Verifies:
  1. Offline timm pretrained tag verification ('vit_small_patch14_dinov2.lvd142m') & native timm overlay --weights loader
  2. Config-driven setup (seed, pos_weights from config.yaml) and B=2 batch construction with strict gold hard labels
  3. DINOv2-Small SlotHead with per-finding linear heads & learned slot embeddings
  4. Gating equivalence: masked 6-slot cross-attention matches unmasked attention over gathered present slots (diff = 0.0)
  5. Path parity: need_weights=True and fused need_weights=False produce identical logits
  6. Non-finite pixel safety & all-absent sequence guard (train, eval, and backward grad verification)
  7. MaskedBCEWithLogitsLoss with continuous pos_weight interpolation verified against closed-form reference
     with diverse soft targets (0.22, 0.90, 0.99), plus branchless exact 0.0 loss & zero grad on all-NaN target rows
  8. Disjoint parameter group partitioning with weight-decay exclusions, comprehensive isfinite gradient check,
     weight mutation, and tight differential learning rate ratio verification (25-45x on block -1 fc1 vs head)
  9. CUDA FP16 AMP forward+backward with calibrated init_scale, unscaled isfinite check, synchronized timing,
     and isolated realistic per-GPU batch (B=4) peak memory tracking
 10. Clean telemetry logging with Model Graph tracing (writer.add_graph) and event file verification
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import shutil
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from src.training.losses import MaskedBCEWithLogitsLoss


def seed_everything(seed: int = 42) -> None:
    """Set random seeds across stdlib, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DINOv2SlotHeadStub(nn.Module):
    """DINOv2-Small backbone + 6-Slot SlotHead cross-attention module.

    Features:
      - Per-finding linear heads: (attended * cls_w).sum(-1) + cls_b prevents logit collapse
      - Learned slot embeddings (6, D) provide explicit sequence identity to keys
      - Input & token zero-masking prevents NaN/Inf pixels in absent slots from poisoning output
      - All-slots-absent guard: key_padding_mask & ~all_masked prevents all-inf softmax NaNs
      - LayerNorm residual for attention stability
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_slots: int = 6,
        num_findings: int = 12,
        num_heads: int = 6,
        img_size: int = 336,
        weights_path: str | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.num_findings = num_findings
        self.img_size = img_size

        if weights_path is not None:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Specified offline weights file not found: {weights_path}")
            # Load offline weights using native timm overlay with automatic bicubic pos_embed resampling
            self.backbone = timm.create_model(
                "vit_small_patch14_dinov2.lvd142m",
                pretrained=True,
                pretrained_cfg_overlay=dict(file=weights_path),
                num_classes=0,
                img_size=img_size,
            )
        else:
            self.backbone = timm.create_model(
                "vit_small_patch14_dinov2",
                pretrained=False,
                num_classes=0,
                img_size=img_size,
            )

        # Learned slot identity embeddings (6, D)
        self.slot_embed = nn.Parameter(torch.randn(num_slots, embed_dim) * 0.02)

        # 12 finding-specific learnable query tokens (12, D)
        self.finding_queries = nn.Parameter(torch.randn(num_findings, embed_dim) * 0.02)

        # Cross-attention: findings query contrast slots
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # LayerNorm & residual
        self.norm = nn.LayerNorm(embed_dim)

        # Per-finding linear classification head (12, D) + bias (12,)
        self.cls_w = nn.Parameter(torch.randn(num_findings, embed_dim) * 0.02)
        self.cls_b = nn.Parameter(torch.zeros(num_findings))

    def forward(
        self,
        x: torch.Tensor,
        presence_mask: torch.Tensor,
        need_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, 6, 3, H, W)
            presence_mask: (B, 6) binary sequence indicator
            need_weights: If True, also returns attention weights (B, 12, 6)

        Returns:
            Raw logits of shape (B, 12). If need_weights is True, returns (logits, attn_weights).
        """
        b, s, c, h, w = x.shape

        # Step A: Input pixel safety — zero absent slot pixels before backbone to block NaNs/Infs
        mask_5d = presence_mask.bool().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x_safe = torch.where(mask_5d, x, torch.zeros_like(x))

        # Step B: Backbone forward pass
        x_flat = x_safe.reshape(b * s, c, h, w)
        slot_tokens = self.backbone(x_flat).reshape(b, s, self.embed_dim)

        # Step C: Token safety — zero absent tokens and inject slot identity
        slot_tokens = torch.where(
            presence_mask.bool().unsqueeze(-1),
            slot_tokens + self.slot_embed.unsqueeze(0),
            torch.zeros_like(slot_tokens),
        )

        # Step D: Cross-attention queries & masks
        queries = self.finding_queries.unsqueeze(0).expand(b, -1, -1)
        key_padding_mask = ~(presence_mask.bool())

        # Guard: if all slots in a study are absent, attend uniformly over zeros rather than NaN
        all_masked = key_padding_mask.all(dim=-1, keepdim=True)
        safe_key_padding_mask = key_padding_mask & ~all_masked

        attended, attn_weights = self.cross_attn(
            query=queries,
            key=slot_tokens,
            value=slot_tokens,
            key_padding_mask=safe_key_padding_mask,
            need_weights=need_weights,
        )

        # Step E: Residual + LayerNorm
        attended = self.norm(queries + attended)

        # Step F: Per-finding linear projections
        logits = (attended * self.cls_w).sum(dim=-1) + self.cls_b

        if need_weights:
            return logits, attn_weights
        return logits


class DINOv2GraphWrapper(nn.Module):
    """Wrapper that returns strictly a single tensor for robust TensorBoard graph tracing."""

    def __init__(self, model: DINOv2SlotHeadStub) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, presence_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(x, presence_mask, need_weights=False)
        return out if isinstance(out, torch.Tensor) else out[0]


def get_synthetic_batch(device: torch.device, img_size: int = 336):
    """Generate a B=2 test batch exercising pseudo vs gold targets and clinical silence rules."""
    b, s, c, h, w = 2, 6, 3, img_size, img_size
    images = torch.randn(b, s, c, h, w, dtype=torch.float32, device=device)

    # Study 0: slot 5 absent; Study 1: slot 2 absent
    presence_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    # 12 targets:
    # [ACL, MCL, Med Meniscus, Lat Meniscus, Med OA, Lat OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture]
    targets = torch.tensor(
        [
            # Study 0: Pseudo study. Acute tears silent=NaN, structural silent=0.0,
            # Effusion is 0.0 so Synovitis soft target is 0.22, with diverse soft targets (0.90, 0.99)
            [float("nan"), 0.0, 0.99, float("nan"), 0.0, 0.0, 0.90, 0.0, 0.22, 0.0, float("nan"), 0.0],
            # Study 1: Gold study with strictly hard binary 0/1 labels (no NaNs, no soft values)
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    is_gold = torch.tensor([0, 1], dtype=torch.int64, device=device)
    return images, presence_mask, targets, is_gold


def compute_reference_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    is_gold: torch.Tensor,
    pos_weights: torch.Tensor,
    gold_weight: float = 5.0,
) -> torch.Tensor:
    """Independent manual implementation of MaskedBCEWithLogitsLoss with continuous pos_weight scaling."""
    valid_mask = ~torch.isnan(targets)
    safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))

    bce = F.binary_cross_entropy_with_logits(logits, safe_targets, reduction="none")

    # Sample-level gold upweighting
    weights = torch.ones_like(logits)
    for b in range(logits.shape[0]):
        if is_gold[b]:
            weights[b] = gold_weight

    # Continuous label-level positive class weighting: 1.0 at 0.0, pos_weight at 1.0
    label_weights = 1.0 + (pos_weights.unsqueeze(0) - 1.0) * safe_targets
    weights = weights * label_weights

    weighted = bce * weights * valid_mask.float()
    n_valid = valid_mask.float().sum()
    return weighted.sum() / n_valid.clamp_min(1.0)


def get_parameter_groups(
    model: nn.Module,
    lr_backbone: float = 3e-5,
    lr_head: float = 1e-3,
    weight_decay: float = 0.02,
) -> list[dict]:
    """Build disjoint optimizer parameter groups with explicit weight decay exclusions."""
    backbone_decay, backbone_no_decay = [], []
    head_decay, head_no_decay = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = name.startswith("backbone.")
        is_no_decay = (
            p.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name
            or "pos_embed" in name
            or "cls_token" in name
            or "slot_embed" in name
            or "finding_queries" in name
        )
        if is_backbone:
            (backbone_no_decay if is_no_decay else backbone_decay).append(p)
        else:
            (head_no_decay if is_no_decay else head_decay).append(p)

    groups = [
        {"params": backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        {"params": head_decay, "lr": lr_head, "weight_decay": weight_decay},
        {"params": head_no_decay, "lr": lr_head, "weight_decay": 0.0},
    ]

    total_grouped = sum(len(g["params"]) for g in groups)
    total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert total_grouped == total_trainable, f"Grouped {total_grouped} params, expected {total_trainable}"
    unique_params = set(id(p) for g in groups for p in g["params"])
    assert len(unique_params) == total_trainable, "Duplicate parameters across optimizer groups!"

    return groups


def log_smoke_telemetry(
    log_dir: str,
    tag: str,
    value: float,
    model: DINOv2SlotHeadStub,
    images: torch.Tensor,
    presence_mask: torch.Tensor,
    require_tb: bool = False,
) -> str:
    """Log scalar and model computational graph to TensorBoard, verifying file creation."""
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=log_dir)
        try:
            # 1. Log scalar
            writer.add_scalar(tag, value, 0)

            # 2. Log Model Graph via dedicated single-output wrapper
            graph_wrapper = DINOv2GraphWrapper(model)
            writer.add_graph(graph_wrapper, (images[:1], presence_mask[:1]))
            writer.flush()
        finally:
            writer.close()

        # Check newly created event file
        events = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
        if not events:
            raise RuntimeError(f"SummaryWriter completed but no event file was found in {log_dir}")
        return "tensorboard.SummaryWriter (scalar + graph verified on disk)"
    except (ImportError, ModuleNotFoundError) as err:
        if require_tb:
            raise RuntimeError(f"TensorBoard is strictly required but not installed: {err}") from err
        event_path = os.path.join(log_dir, "events.smoke.log")
        with open(event_path, "w", encoding="utf-8") as f:
            f.write(f"step=0 {tag}={value:.6f}\n")
        return "fallback.EventLog (warning: tensorboard package not installed)"


def run_smoke_tracer(require_tb: bool = False, tiny: bool = False, weights_path: str | None = None) -> int:
    """Execute end-to-end verification tracer."""
    print("=" * 80)
    print("RSNA Knee MRI: Refined Efficiency Track End-to-End Smoke Tracer (Final)")
    print("=" * 80)

    # 1. Offline timm tag verification & config loading
    timm_cfg = timm.models.get_pretrained_cfg("vit_small_patch14_dinov2.lvd142m")
    assert timm_cfg is not None, "Pretrained config for vit_small_patch14_dinov2.lvd142m is None!"
    print(f"[1/10] Offline timm tag verified ('vit_small_patch14_dinov2.lvd142m', res={timm_cfg.input_size})")

    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("project", {}).get("seed", 42)
    seed_everything(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_size = 112 if tiny else 336

    # 2. Batch construction (B=2)
    images, presence_mask, targets, is_gold = get_synthetic_batch(device, img_size=img_size)
    assert images.shape == (2, 6, 3, img_size, img_size)
    assert presence_mask.shape == (2, 6)
    assert targets.shape == (2, 12)
    assert is_gold.shape == (2,)
    print(f"[2/10] Batch constructed (B=2, seed={seed}, img_size={img_size}, Device={device})")

    # 3. Model forward pass & path parity check
    model = DINOv2SlotHeadStub(img_size=img_size, weights_path=weights_path).to(device)
    model.train()
    logits_fused = model(images, presence_mask, need_weights=False)
    logits_weights, attn_weights = model(images, presence_mask, need_weights=True)

    # Verify both paths produce identical logits
    parity_diff = (logits_fused - logits_weights).abs().max().item()
    assert torch.allclose(logits_fused, logits_weights, atol=1e-5), f"Path parity mismatch: {parity_diff}"
    assert attn_weights.shape == (2, 12, 6)
    print(f"[3/10] Forward verified: logits={logits_fused.shape}, path parity diff={parity_diff:.2e}")

    # 4. Gating Equivalence: Compare masked 6-slot attention against unmasked gathered present slots
    study0_slot5_attn = attn_weights[0, :, 5].max().item()
    study1_slot2_attn = attn_weights[1, :, 2].max().item()
    assert study0_slot5_attn == 0.0, f"Absent slot 5 received non-zero attention: {study0_slot5_attn}"
    assert study1_slot2_attn == 0.0, f"Absent slot 2 received non-zero attention: {study1_slot2_attn}"

    with torch.no_grad():
        b, s, c, h, w = images.shape
        x_flat = images.reshape(b * s, c, h, w)
        slot_tokens = model.backbone(x_flat).reshape(b, s, model.embed_dim) + model.slot_embed.unsqueeze(0)
        queries = model.finding_queries.unsqueeze(0).expand(b, -1, -1)

        # Study 0: slot 5 is absent (first 5 slots present)
        # Compare masked attention over all 6 slots vs unmasked attention over only the 5 present slots
        key_pad_mask = ~(presence_mask[:1].bool())
        all_m = key_pad_mask.all(dim=-1, keepdim=True)
        safe_mask = key_pad_mask & ~all_m

        att_masked, _ = model.cross_attn(queries[:1], slot_tokens[:1], slot_tokens[:1], key_padding_mask=safe_mask)
        att_unmasked, _ = model.cross_attn(queries[:1], slot_tokens[:1, :5], slot_tokens[:1, :5])

        equiv_diff = (att_masked - att_unmasked).abs().max().item()
        assert torch.allclose(att_masked, att_unmasked, atol=1e-5), f"Masked vs unmasked gathered diff: {equiv_diff}"

    print(f"[4/10] Gating mathematically proven: masked 6-slot matches unmasked 5-present slots (diff={equiv_diff:.2e})")

    # 5. Non-finite pixel safety & all-absent sequence guard
    images_nan = images.clone()
    images_nan[0, 5] = float("nan")
    logits_nan = model(images_nan, presence_mask, need_weights=False)
    assert torch.isfinite(logits_nan).all(), "NaN pixels in absent slot poisoned logits!"
    logits_nan.sum().backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN gradient on {name} from absent-slot NaN pixel!"
    model.zero_grad()

    # All-absent sequence guard verified in train, eval, and backward
    all_absent_mask = torch.zeros(1, 6, device=device)
    model.train()
    logits_all_absent_train = model(images[:1], all_absent_mask, need_weights=False)
    assert torch.isfinite(logits_all_absent_train).all(), "All-absent train logits are non-finite!"
    logits_all_absent_train.sum().backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"All-absent backward produced non-finite grad on {name}!"
    model.zero_grad()

    model.eval()
    with torch.no_grad():
        logits_all_absent_eval = model(images[:1], all_absent_mask, need_weights=False)
    assert torch.isfinite(logits_all_absent_eval).all(), "All-absent eval logits are non-finite!"
    print("[5/10] Edge cases verified: absent NaN pixels safe, all-absent guard tested (train/eval/bwd)")

    # 6. Masked BCE Loss mathematically verified against reference
    model.train()
    logits_for_loss = model(images, presence_mask, need_weights=False)
    pos_weights = torch.tensor(cfg["labels"]["pos_weights"], dtype=torch.float32, device=device)
    criterion = MaskedBCEWithLogitsLoss(gold_weight=5.0, pos_weight=pos_weights, continuous_pos_weight=True)

    loss = criterion(logits_for_loss, targets, is_gold)
    expected_loss = compute_reference_loss(logits_for_loss, targets, is_gold, pos_weights, gold_weight=5.0)
    assert torch.isfinite(loss), "Computed loss is non-finite!"
    assert torch.allclose(loss, expected_loss, atol=1e-5), f"Loss mismatch: {loss.item()} vs {expected_loss.item()}"

    # Gradient probe at NaN positions
    logits_probe = logits_for_loss.detach().clone().requires_grad_(True)
    loss_probe = criterion(logits_probe, targets, is_gold)
    loss_probe.backward()
    nan_mask = torch.isnan(targets)
    assert torch.isfinite(logits_probe.grad).all(), "Logits gradient has NaNs/Infs!"
    assert (logits_probe.grad[nan_mask] == 0.0).all(), "Gradient at NaN target positions is non-zero!"
    assert (logits_probe.grad[~nan_mask] != 0.0).all(), "Gradient at valid target positions is zero!"
    print(f"[6/10] Loss mathematically verified: loss={loss.item():.4f}, NaN pos grads == 0.0")

    # 7. All-NaN target row test
    all_nan_targets = torch.full((2, 12), float("nan"), device=device)
    logits_nan_probe = logits_for_loss.detach().clone().requires_grad_(True)
    loss_all_nan = criterion(logits_nan_probe, all_nan_targets, is_gold)
    loss_all_nan.backward()
    assert loss_all_nan.item() == 0.0, f"Expected 0.0 loss for all-NaN batch, got {loss_all_nan.item()}"
    assert torch.isfinite(logits_nan_probe.grad).all() and (logits_nan_probe.grad == 0.0).all(), "All-NaN grads non-zero!"
    print(f"[7/10] All-NaN target test: returns {loss_all_nan.item():.2f} with exact 0.0 gradients")

    # 8. Differential AdamW & Disjoint Parameter Group Verification
    param_groups = get_parameter_groups(model, lr_backbone=3e-5, lr_head=1e-3, weight_decay=0.02)
    optimizer = torch.optim.AdamW(param_groups)
    optimizer.zero_grad()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                raise RuntimeError(f"Parameter '{name}' gradient is None!")
            if not torch.isfinite(param.grad).all():
                raise RuntimeError(f"Parameter '{name}' gradient contains NaN or Inf!")
            if not (param.grad != 0).any():
                raise RuntimeError(f"Parameter '{name}' gradient is all zeros!")

    cls_w_before = model.cls_w.clone()
    bb_w_before = model.backbone.blocks[-1].mlp.fc1.weight.clone()

    optimizer.step()

    assert not torch.equal(cls_w_before, model.cls_w), "Classifier weights failed to mutate!"
    assert not torch.equal(bb_w_before, model.backbone.blocks[-1].mlp.fc1.weight), "Backbone weights failed to mutate!"
    assert torch.isfinite(model.cls_w).all() and torch.isfinite(model.backbone.blocks[-1].mlp.fc1.weight).all()

    # Differential learning rate verification (tight check on block -1 fc1 vs head: 25-45x)
    head_delta = (model.cls_w - cls_w_before).abs().median().item()
    bb_delta = (model.backbone.blocks[-1].mlp.fc1.weight - bb_w_before).abs().median().item()
    update_ratio = head_delta / max(bb_delta, 1e-8)
    assert 25.0 <= update_ratio <= 45.0, f"Update ratio out of expected range (25-45x): {update_ratio:.2f}"
    print(f"[8/10] Optimizer verified: disjoint groups, all grads finite, update ratio={update_ratio:.1f}x (1e-3 vs 3e-5)")

    # 9. CUDA FP16 AMP / Isolated Peak Memory & Synchronized Timing
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model.train()
        scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)

        # Run 2 training steps to verify stable scaling
        for step in range(2):
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                amp_logits = model(images, presence_mask, need_weights=False)
                amp_loss = criterion(amp_logits, targets, is_gold)
            assert torch.isfinite(amp_loss), "CUDA FP16 loss is non-finite!"
            scaler.scale(amp_loss).backward()
            scaler.unscale_(optimizer)

            for name, p in model.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), f"Unscaled FP16 gradient non-finite on {name}!"
            scaler.step(optimizer)
            scaler.update()

        assert scaler.get_scale() >= 1.0, f"Scaler underflowed: scale={scaler.get_scale()}"

        # Measure memory at realistic per-GPU training batch: B=4 (24 images)
        torch.cuda.reset_peak_memory_stats(device)
        batch_4_img = torch.randn(4, 6, 3, img_size, img_size, device=device)
        batch_4_mask = torch.ones(4, 6, device=device)
        batch_4_targets = torch.zeros(4, 12, device=device)
        batch_4_gold = torch.zeros(4, dtype=torch.int64, device=device)
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            b4_logits = model(batch_4_img, batch_4_mask, need_weights=False)
            b4_loss = criterion(b4_logits, batch_4_targets, batch_4_gold)
        scaler.scale(b4_loss).backward()
        peak_train_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        # Pure FP16 eval timing with warmup
        model.eval()
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16):
                for _ in range(3):
                    _ = model(images, presence_mask, need_weights=False)
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(10):
                    _ = model(images, presence_mask, need_weights=False)
                torch.cuda.synchronize()

        sec_per_study = (time.time() - t0) / 20.0
        print(f"[9/10] CUDA FP16 verified: unscaled grads finite, eval latency={sec_per_study*1000:.1f}ms/study, peak train mem (B=4)={peak_train_mem_mb:.1f}MB")
    else:
        model.eval()
        t0 = time.time()
        with torch.no_grad():
            _ = model(images, presence_mask, need_weights=False)
        dt = time.time() - t0
        sec_per_study = dt / images.shape[0]
        print(f"[9/10] Benchmark on CPU: latency={sec_per_study:.2f}s/study")

    # 10. Clean Telemetry Logging (Model Graph + Scalar)
    smoke_dir = str(PROJECT_ROOT / "outputs" / "tensorboard" / "smoke" / "test_run")
    backend = log_smoke_telemetry(
        smoke_dir,
        "smoke/refined_loss",
        loss.item(),
        model,
        images,
        presence_mask,
        require_tb=require_tb,
    )
    print(f"[10/10] Telemetry verified: logged to {smoke_dir} via {backend}")

    print("=" * 80)
    print("SUCCESS: Refined End-to-End Wiring Smoke Tracer passed all verifications!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Wiring Smoke Tracer")
    parser.add_argument("--require-tensorboard", action="store_true", help="Fail if tensorboard is not installed")
    parser.add_argument("--tiny", action="store_true", help="Use 112px images for ultra-fast local checks")
    parser.add_argument("--weights", type=str, default=None, help="Path to offline DINOv2 weights checkpoint")
    args = parser.parse_args()
    sys.exit(run_smoke_tracer(require_tb=args.require_tensorboard, tiny=args.tiny, weights_path=args.weights))
