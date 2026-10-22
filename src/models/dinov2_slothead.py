# filepath: src/models/dinov2_slothead.py
"""src/models/dinov2_slothead.py

DINOv2-Small backbone + 18-token (6 slots x 3 depth zones) SlotHead cross-attention
architecture for RSNA Knee Abnormality Detection (Efficiency Track).
Single-Responsibility Module: purely defines the neural network architecture.
"""

from __future__ import annotations

import logging
import os
import timm
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

NUM_SLOTS, NUM_ZONES = 6, 3
TOTAL_TOKENS = NUM_SLOTS * NUM_ZONES
NUM_CLASSES = 12


class DINOv2SlotHead(nn.Module):
    """DINOv2-Small with 18-token (6 slots x 3 depth zones) SlotHead cross-attention."""

    def __init__(
        self,
        embed_dim: int = 384,
        num_slots: int = NUM_SLOTS,
        num_findings: int = NUM_CLASSES,
        num_heads: int = 6,
        img_size: int = 336,
        pretrained: bool = False,
        weights_path: str | None = None,
        slot_dropout_p: float = 0.0,
        attn_dropout_p: float = 0.2,
        pre_cls_dropout_p: float = 0.3,
        init_bias_priors: list[float] | torch.Tensor | None = None,
        use_grad_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.num_findings = num_findings
        self.num_heads = num_heads
        self.img_size = img_size
        self.slot_dropout_p = slot_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.pre_cls_dropout_p = pre_cls_dropout_p
        self.use_grad_checkpointing = use_grad_checkpointing

        # Backbone: DINOv2-Small patch14
        if weights_path is not None:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Offline weights not found: {weights_path}")
            self.backbone = timm.create_model(
                "vit_small_patch14_dinov2.lvd142m",
                pretrained=True,
                pretrained_cfg_overlay=dict(file=weights_path),
                num_classes=0,
                img_size=img_size,
            )
        else:
            self.backbone = timm.create_model(
                "vit_small_patch14_dinov2.lvd142m",
                pretrained=pretrained,
                num_classes=0,
                img_size=img_size,
            )

        expected_tokens = 1 + (img_size // 14) ** 2
        if self.backbone.pos_embed.shape[1] != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} pos_embed tokens for img_size={img_size}, "
                f"got {self.backbone.pos_embed.shape[1]}"
            )

        # Slot and Zone identity embeddings (factored additive, std=0.20)
        self.slot_embed = nn.Parameter(torch.randn(num_slots, embed_dim) * 0.20)
        self.zone_embed = nn.Parameter(torch.randn(NUM_ZONES, embed_dim) * 0.20)

        # 12 finding-specific query vectors
        self.finding_queries = nn.Parameter(torch.randn(num_findings, embed_dim) * 0.02)

        # Multi-head cross-attention over 18 slot-zone tokens (default: 6 heads per spec)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout_p,
            batch_first=True,
        )

        # Residual LayerNorm: LayerNorm(Q + Attn)
        self.norm = nn.LayerNorm(embed_dim)

        # Pre-classifier dropout regularization
        self.pre_cls_dropout = nn.Dropout(p=pre_cls_dropout_p)

        # Per-finding linear classification head (12, D)
        self.cls_w = nn.Parameter(torch.randn(num_findings, embed_dim) * 0.02)

        if init_bias_priors is not None:
            priors = torch.as_tensor(init_bias_priors, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
            self.cls_b = nn.Parameter(torch.log(priors / (1.0 - priors)))
        else:
            self.cls_b = nn.Parameter(torch.zeros(num_findings))

    def _apply_slot_dropout(self, present: torch.Tensor) -> torch.Tensor:
        if not self.training or self.slot_dropout_p <= 0.0:
            return present
        keep_mask = torch.rand_like(present, dtype=torch.float32) >= self.slot_dropout_p
        dropped = present & keep_mask
        return torch.where(dropped.any(dim=-1, keepdim=True), dropped, present)

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        ckpt = self.use_grad_checkpointing and self.training and torch.is_grad_enabled()
        if not ckpt:
            return self.backbone(x)

        bb = self.backbone
        x = bb.patch_embed(x)
        x = bb._pos_embed(x)
        x = bb.patch_drop(x)
        x = bb.norm_pre(x)
        for blk in bb.blocks:
            x = checkpoint(blk, x, use_reentrant=False)
        x = bb.norm(x)
        return x[:, 0]

    def _forward_backbone_and_head(
        self,
        x: torch.Tensor,
        presence_mask: torch.Tensor,
        need_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b = x.shape[0]
        if x.ndim == 5:
            if x.shape[1] != NUM_SLOTS or x.shape[2] != 9:
                raise RuntimeError(f"Expected (B, {NUM_SLOTS}, 9, H, W), got {tuple(x.shape)}")
        elif x.ndim == 6:
            if tuple(x.shape[1:4]) != (NUM_SLOTS, NUM_ZONES, 3):
                raise RuntimeError(f"Expected (B, {NUM_SLOTS}, {NUM_ZONES}, 3, H, W), got {tuple(x.shape)}")
        else:
            raise RuntimeError(f"Unsupported input rank {x.ndim}")

        h_, w_ = x.shape[-2], x.shape[-1]
        x_18 = x.reshape(b, TOTAL_TOKENS, 3, h_, w_)

        presence = presence_mask.to(x.device).float()
        active = presence.unsqueeze(2).expand(b, NUM_SLOTS, NUM_ZONES).reshape(b, TOTAL_TOKENS).clone()

        if self.training and self.slot_dropout_p > 0.0:
            present_b = presence > 0.5
            kept_slots = self._apply_slot_dropout(present_b)
            keep = kept_slots.float().unsqueeze(2).expand(b, NUM_SLOTS, NUM_ZONES).reshape(b, TOTAL_TOKENS)
            active = active * keep

        act_flat = active.reshape(-1) > 0.5
        token_feats = torch.zeros(b * TOTAL_TOKENS, self.embed_dim, device=x.device, dtype=torch.float32)

        if act_flat.any():
            feats = self._forward_backbone(x_18.reshape(b * TOTAL_TOKENS, 3, h_, w_)[act_flat])
            token_feats[act_flat] = feats.float()
        else:
            dummy = self._forward_backbone(x.new_zeros(1, 3, h_, w_)).float()
            token_feats = token_feats + 0.0 * dummy.sum()

        tokens = token_feats.view(b, TOTAL_TOKENS, self.embed_dim)

        slot_idx = torch.arange(NUM_SLOTS, device=x.device).repeat_interleave(NUM_ZONES)
        zone_idx = torch.arange(NUM_ZONES, device=x.device).repeat(NUM_SLOTS)
        tokens = tokens + (self.slot_embed[slot_idx] + self.zone_embed[zone_idx]).unsqueeze(0)

        key_padding_mask = active <= 0.5
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False

        q = self.finding_queries.unsqueeze(0).expand(b, -1, -1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            attn_out, attn_w = self.cross_attn(
                q.float(),
                tokens.float(),
                tokens.float(),
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                average_attn_weights=False,
            )
            a = self.norm(q + attn_out)
            a = self.pre_cls_dropout(a)
            logits = (a * self.cls_w.unsqueeze(0)).sum(dim=-1) + self.cls_b.unsqueeze(0)
            logits = logits.float()

        clean_weights = None
        if need_weights and attn_w is not None:
            clean_weights = attn_w.mean(dim=1)
            clean_weights = torch.where(
                active.unsqueeze(1) > 0.5,
                clean_weights,
                torch.zeros_like(clean_weights),
            )

        return logits, clean_weights

    def forward(self, x: torch.Tensor, presence_mask: torch.Tensor) -> torch.Tensor:
        logits, _ = self._forward_backbone_and_head(x, presence_mask, need_weights=False)
        return logits

    def forward_with_attention(
        self,
        x: torch.Tensor,
        presence_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward_backbone_and_head(x, presence_mask, need_weights=True)
