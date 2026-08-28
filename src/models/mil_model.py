"""
mil_model.py -- 2.5D Multi-Instance Learning Model for RSNA Knee MRI

Architecture:
    - EfficientNet-B2 backbone (timm) with in_chans=stack_size (2.5D input)
    - Gradient checkpointing (~60% VRAM reduction) for Kaggle T4/P100
    - Plane-aware learned attention pooling (separate per anatomical plane)
    - Plane identity encoded via nn.Embedding (not hard-coded bias)
    - plane_mask in forward(): missing planes zero-masked before cross-plane fusion
    - Cross-plane MLP fusion -> raw logit classifier head

Output convention:
    ALWAYS raw logits. NEVER sigmoid inside this module.
    Apply sigmoid only at inference time and AUC computation.

Offline Kaggle:
    Use local_weights_path to load pre-downloaded backbone weights.
    pretrained=True will fail in no-internet Kaggle submissions.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import timm


# ============================================================================
# Attention Pooling Module
# ============================================================================

class AttentionPooling(nn.Module):
    """Learned attention pooling for multi-instance learning (Ilse et al., 2018).

    Aggregates a bag of instance-level feature vectors into a single
    study-level representation via learned attention weights.

    Args:
        embed_dim: Feature vector dimension.
        hidden_dim: Attention MLP hidden layer dimension.
        dropout: Dropout on attention weights before softmax.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool bag of instance features.

        Args:
            x: (B, N_instances, embed_dim)

        Returns:
            pooled: (B, embed_dim) -- study-level representation
            attn_weights: (B, N_instances) -- attention weights (sum to 1)
        """
        attn_logits  = self.attention(x).squeeze(-1)       # (B, N)
        attn_weights = torch.softmax(attn_logits, dim=-1)  # (B, N)
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (B, D)
        return pooled, attn_weights


# ============================================================================
# Main MIL Model
# ============================================================================

class KneeMILModel(nn.Module):
    """2.5D Multi-Instance Learning model for RSNA Knee MRI abnormality detection.

    Processes multi-plane (sagittal, coronal, axial) multi-slice knee MRI studies.
    Each plane is encoded by a shared backbone, pooled via learned attention,
    then fused across planes into a 12-label classifier.

    Args:
        backbone_name: timm model identifier (default: 'efficientnet_b2').
        n_classes: Output labels (12).
        n_planes: Anatomical planes (3: sagittal, coronal, axial).
        stack_size: 2.5D input channels -- consecutive slices (default: 3).
        pretrained: Load ImageNet weights (set False for offline Kaggle).
        local_weights_path: Path to local .pth backbone weights file.
                            If provided, overrides pretrained flag.
        attn_hidden_dim: Attention MLP hidden dim.
        dropout: Dropout before classifier head.
        freeze_backbone_epochs: Epochs to keep backbone frozen during warm-up.
        use_grad_checkpointing: Enable timm gradient checkpointing (~60% VRAM).

    Forward input:
        planes: dict{'sagittal','coronal','axial'} each (B, N, stack_size, H, W)
        plane_mask: (B, 3) bool -- True if plane is present for study.
                    Missing planes (False) get zero-masked pooled embedding.

    Forward output:
        logits: (B, n_classes) raw logits. Do NOT apply sigmoid here.
        attn_weights: dict of per-plane (B, N_slices) attention weights.
    """

    PLANE_NAMES = ["sagittal", "coronal", "axial"]

    def __init__(
        self,
        backbone_name: str = "efficientnet_b2",
        n_classes: int = 12,
        n_planes: int = 3,
        stack_size: int = 3,
        pretrained: bool = True,
        local_weights_path: str | None = None,
        attn_hidden_dim: int = 128,
        dropout: float = 0.3,
        freeze_backbone_epochs: int = 0,
        use_grad_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.n_planes    = n_planes
        self.stack_size  = stack_size
        self.n_classes   = n_classes
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self._current_epoch = 0

        # Backbone: pretrained=False if using local weights
        _pretrained = pretrained and local_weights_path is None
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=_pretrained,
            num_classes=0,        # Remove classification head
            in_chans=stack_size,  # 2.5D: 3 consecutive slices as channels
            global_pool="avg",
        )

        # Load local backbone weights (offline Kaggle)
        if local_weights_path is not None:
            p = Path(local_weights_path)
            if not p.exists():
                raise FileNotFoundError(f"Local backbone weights not found: {p}")
            state = torch.load(p, map_location="cpu")
            # Handle both raw state_dict and checkpoint dicts
            sd = state.get("model_state_dict", state)
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            if missing:
                print(f"     Missing backbone keys: {missing[:5]}...")
            print(f"    Loaded backbone weights from {p}")

        # Gradient checkpointing: ~60% VRAM reduction for T4/P100
        if use_grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)
            print("    Gradient checkpointing enabled")

        embed_dim = self.backbone.num_features

        # Plane identity encoding (separate embedding per plane)
        self.plane_embedding = nn.Embedding(n_planes, embed_dim)

        # Separate attention pooler per plane (planes have distinct anatomy)
        self.plane_poolers = nn.ModuleList([
            AttentionPooling(embed_dim, attn_hidden_dim, dropout=0.1)
            for _ in range(n_planes)
        ])

        # Cross-plane fusion MLP
        fusion_dim = embed_dim * n_planes
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Classification head: raw logits, NO sigmoid
        self.classifier = nn.Linear(fusion_dim // 4, n_classes)

        if freeze_backbone_epochs > 0:
            self._set_backbone_frozen(True)

    def _set_backbone_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = not frozen
        print(f"   Backbone {'FROZEN' if frozen else 'UNFROZEN'}")

    def on_epoch_start(self, epoch: int) -> None:
        """Call at the start of each epoch to manage backbone warm-up freeze."""
        self._current_epoch = epoch
        if self.freeze_backbone_epochs > 0 and epoch == self.freeze_backbone_epochs:
            self._set_backbone_frozen(False)

    def _encode_plane(
        self,
        plane_tensor: torch.Tensor,  # (B, N, stack_size, H, W)
        plane_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode all slices of one plane through the shared backbone.

        Returns:
            pooled: (B, embed_dim) -- attention-pooled plane embedding
            attn_weights: (B, N_slices) -- per-slice attention weights
        """
        B, N, C, H, W = plane_tensor.shape

        # Encode all N slices in one batch: (B*N, C, H, W) -> (B*N, embed_dim)
        features = self.backbone(
            plane_tensor.view(B * N, C, H, W)
        ).view(B, N, -1)  # (B, N, embed_dim)

        # Add plane identity embedding (broadcasts over N slices)
        plane_emb = self.plane_embedding(
            torch.full((B,), plane_idx, dtype=torch.long, device=plane_tensor.device)
        )  # (B, embed_dim)
        features = features + plane_emb.unsqueeze(1)  # (B, N, embed_dim)

        return self.plane_poolers[plane_idx](features)

    def forward(
        self,
        planes: dict[str, torch.Tensor],
        plane_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass through the full MIL model.

        Args:
            planes: {name: tensor} for 'sagittal', 'coronal', 'axial'.
                    Each tensor: (B, N_slices, stack_size, H, W).
            plane_mask: (B, 3) bool. True = plane present for that study.
                        Missing planes are zero-masked in fused embedding
                        (not softmax-pooled, which would generate noise from
                        zero-valued backbone features).

        Returns:
            logits: (B, n_classes) raw logits. Apply sigmoid at inference only.
            attn_weights: dict with per-plane (B, N_slices) attention tensors.
        """
        plane_features = []
        attn_dict = {}

        for plane_idx, plane_name in enumerate(self.PLANE_NAMES):
            pooled, attn = self._encode_plane(planes[plane_name], plane_idx)
            attn_dict[plane_name] = attn

            # Zero-mask missing planes (plane_mask=False -> zero embedding)
            if plane_mask is not None:
                mask = plane_mask[:, plane_idx].float().unsqueeze(1)  # (B, 1)
                pooled = pooled * mask

            plane_features.append(pooled)

        # Fuse across planes
        fused  = torch.cat(plane_features, dim=-1)  # (B, embed_dim * n_planes)
        fused  = self.fusion_mlp(fused)
        logits = self.classifier(fused)              # (B, n_classes) raw logits

        return logits, attn_dict
