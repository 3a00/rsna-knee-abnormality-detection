"""src/models/dinov2_slothead.py

DINOv2-Small backbone + 6-Slot SlotHead cross-attention architecture
for RSNA Knee Abnormality Detection (Efficiency Track).

Single-Responsibility Module: purely defines the neural network architecture.
"""

from __future__ import annotations

import os

import timm
import torch
import torch.nn as nn


class DINOv2SlotHead(nn.Module):
    """DINOv2-Small vision transformer with SlotHead cross-attention pooling.

    Key Architectural Principles:
      1. Present-Slot Gathering: Only present sequences are forwarded through DINOv2,
         saving FLOPs proportional to absent sequence frequency (slots 3-5 absent in ~40-60% of studies).
      2. Unconditional FP32 SlotHead: Cross-attention, LayerNorm, and linear heads run
         strictly in FP32 with autocast disabled, guaranteeing mathematical immunity
         to FP16 overflow without reactive GPU-to-CPU sync bubbles.
      3. DDP Gradient Guard: All-absent batches execute a dummy 1-slice forward pass so
         backbone parameters always participate in gradient synchronization across ranks.
      4. Label Prior Bias Initialization: Initializes cls_b to log(p / (1 - p)) to stabilize
         early training on highly imbalanced finding distributions.
      5. Slot Dropout: Randomly drops present slots during training (slot_dropout_p=0.10-0.15
         configured in training recipe; default is 0.0) to build robustness against missing sequences.
      6. Clean Return Typing: Standard forward() returns strictly raw logits of shape (B, 12).
         Diagnostic forward_with_attention() returns (logits, attn_weights).
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_slots: int = 6,
        num_findings: int = 12,
        num_heads: int = 6,
        img_size: int = 336,
        pretrained: bool = False,
        weights_path: str | None = None,
        slot_dropout_p: float = 0.0,
        init_bias_priors: list[float] | torch.Tensor | None = None,
    ) -> None:
        """Initializes the DINOv2SlotHead model.

        Args:
            embed_dim: Latent feature dimension (default: 384 for DINOv2-Small).
            num_slots: Number of anatomical contrast sequence slots (default: 6).
            num_findings: Number of binary clinical abnormality targets (default: 12).
            num_heads: Multi-head attention heads count (default: 6).
            img_size: Input spatial dimension in pixels (default: 336).
            pretrained: Whether to download pretrained weights via timm (default: False for offline compliance).
            weights_path: Path to offline .pth checkpoint for weight loading and pos_embed resampling.
            slot_dropout_p: Probability of dropping present slots during training (default: 0.0).
            init_bias_priors: Empirical positive label priors for zero-step BCE log-odds bias initialization.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.num_findings = num_findings
        self.num_heads = num_heads
        self.img_size = img_size
        self.slot_dropout_p = slot_dropout_p

        # Backbone: DINOv2-Small patch14 (consistent .lvd142m tag)
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

        # Slot identity embeddings: E_slot in R^(6 x D)
        # Initialized with std=0.20 (~20% of DINOv2 LayerNorm feature std ~1.0, norm ~19.6)
        self.slot_embed = nn.Parameter(torch.randn(num_slots, embed_dim) * 0.20)

        # 12 finding-specific query vectors: Q in R^(12 x D)
        self.finding_queries = nn.Parameter(torch.randn(num_findings, embed_dim) * 0.02)

        # Multi-head cross-attention: findings query contrast slots
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Residual LayerNorm
        self.norm = nn.LayerNorm(embed_dim)

        # Per-finding linear classification head (12, D)
        self.cls_w = nn.Parameter(torch.randn(num_findings, embed_dim) * 0.02)

        # Initialize cls_b from prior probabilities or log-odds
        # Note: under loss weighting w_f = 1 + (pos_weight - 1) * y, the theoretical optimal bias
        # shifts by ~log(pos_weight), but empirical log(p / (1 - p)) provides stable zero-step BCE init.
        if init_bias_priors is not None:
            priors = torch.as_tensor(init_bias_priors, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
            self.cls_b = nn.Parameter(torch.log(priors / (1.0 - priors)))
        else:
            self.cls_b = nn.Parameter(torch.zeros(num_findings))

    def _apply_slot_dropout(self, present: torch.Tensor) -> torch.Tensor:
        """Randomly drop present slots during training, restoring original slots if all get dropped.

        Args:
            present: Boolean tensor of active sequences of shape (B, S).

        Returns:
            Stochastically masked boolean tensor of shape (B, S).
        """
        if not self.training or self.slot_dropout_p <= 0.0:
            return present
        keep_mask = torch.rand_like(present, dtype=torch.float32) >= self.slot_dropout_p
        dropped = present & keep_mask
        # Guarantee at least one slot remains active per study if originally present
        return torch.where(dropped.any(dim=-1, keepdim=True), dropped, present)

    def _forward_backbone_and_head(
        self,
        x: torch.Tensor,
        presence_mask: torch.Tensor,
        need_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, s, c, h, w = x.shape
        present = presence_mask.bool()

        # Slot dropout during training
        if self.training and self.slot_dropout_p > 0.0:
            present = self._apply_slot_dropout(present)

        # Step 1: Gather and forward ONLY present slots through the ViT backbone
        feats = x.new_zeros(b, s, self.embed_dim, dtype=torch.float32)
        if present.any():
            feats[present] = self.backbone(x[present]).float()
        else:
            # DDP parameter synchronization guard: if entire batch is absent,
            # run a dummy slice through backbone so parameters always receive gradients
            dummy = self.backbone(x.new_zeros(1, c, h, w)).float()
            feats = feats + 0.0 * dummy.sum()

        # Step 2: Unconditional FP32 SlotHead Execution (branch-free, zero GPU sync)
        with torch.autocast(device_type=x.device.type, enabled=False):
            # Inject slot identity embeddings and gate absent tokens
            tokens = (feats + self.slot_embed.unsqueeze(0)) * present.unsqueeze(-1).float()

            key_padding_mask = ~present
            all_absent = key_padding_mask.all(dim=-1, keepdim=True)
            safe_key_padding_mask = key_padding_mask & ~all_absent

            queries = self.finding_queries.unsqueeze(0).expand(b, -1, -1)
            attended, attn_weights = self.cross_attn(
                query=queries,
                key=tokens,
                value=tokens,
                key_padding_mask=safe_key_padding_mask,
                need_weights=need_weights,
            )

            attended = self.norm(queries + attended)
            logits = (attended * self.cls_w).sum(dim=-1) + self.cls_b

        clean_weights = None
        if need_weights and attn_weights is not None:
            clean_weights = torch.where(
                present.unsqueeze(1),
                attn_weights,
                torch.zeros_like(attn_weights),
            )

        return logits, clean_weights

    def forward(self, x: torch.Tensor, presence_mask: torch.Tensor) -> torch.Tensor:
        """Standard model forward pass returning strictly raw logits of shape (B, 12).

        Args:
            x: Input batch tensor of shape (B, 6, 3, H, W).
            presence_mask: Binary mask of active slots of shape (B, 6).

        Returns:
            Raw unscaled logits tensor of shape (B, 12) in FP32.
        """
        logits, _ = self._forward_backbone_and_head(x, presence_mask, need_weights=False)
        return logits

    def forward_with_attention(
        self,
        x: torch.Tensor,
        presence_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Diagnostic forward pass returning raw logits and head-averaged attention weights.

        Args:
            x: Input batch tensor of shape (B, 6, 3, H, W).
            presence_mask: Binary mask of active slots of shape (B, 6).

        Returns:
            Tuple of:
                logits: Raw unscaled logits of shape (B, 12).
                attn_weights: Head-averaged slot attention weights of shape (B, 12, 6).
        """
        logits, attn_weights = self._forward_backbone_and_head(x, presence_mask, need_weights=True)
        assert attn_weights is not None
        return logits, attn_weights
