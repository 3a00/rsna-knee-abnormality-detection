"""
losses.py -- Loss Functions for RSNA Knee MRI Training

MaskedBCEWithLogitsLoss:
    BCE with NaN masking for the three-state label scheme:
        1   -> included (target=1.0)
        0   -> included (target=0.0)
        NaN -> excluded (silence / unaddressed -- never replaced with 0)
    Plus gold study upweighting and optional per-label pos_weight.

PlattScaler:
    Post-training temperature calibration.
    MUST be fitted on Out-of-Fold (OOF) gold validation logits -- not
    in-sample training logits (which would cause overfitting).

Core rules:
    - BCEWithLogitsLoss throughout (numerically stable on raw logits)
    - sigmoid() NEVER called inside either class -- always on raw logits
    - NaN targets NEVER replaced with 0 -- masked via valid_mask
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBCEWithLogitsLoss(nn.Module):
    """BCE loss with NaN masking and gold study upweighting.

    Handles the three-state label scheme from Phase 2 pseudo_labels.csv:
        1   -> positive, included in loss
        0   -> negative, included in loss
        NaN -> unaddressed silence, EXCLUDED from loss (never contributes gradient)

    Gold upweighting: gold studies (is_gold=1, n=58) receive gold_weight x
    higher loss contribution to compensate for their 1.3% share in the dataset.

    Args:
        gold_weight: Loss multiplier for gold studies (default: 5.0).
                     Effectively brings gold studies to ~6.2% of gradient signal.
        pos_weight: Per-label positive class weight tensor (12,) for
                    class imbalance correction (optional).
        reduction: 'mean' (default) or 'sum'.
    """

    def __init__(
        self,
        gold_weight: float = 5.0,
        pos_weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gold_weight = gold_weight
        self.pos_weight  = pos_weight
        self.reduction   = reduction

    def forward(
        self,
        logits: torch.Tensor,   # (B, 12) raw logits
        targets: torch.Tensor,  # (B, 12) float -- NaN where label is unaddressed
        is_gold: torch.Tensor,  # (B,) int -- 1 for gold studies, 0 for pseudo
    ) -> torch.Tensor:
        """Compute masked, gold-upweighted BCE loss.

        Args:
            logits: (B, 12) raw logits. NEVER sigmoid-activated.
            targets: (B, 12) with NaN for unaddressed silence.
            is_gold: (B,) gold indicator.

        Returns:
            Scalar loss tensor.
        """
        # Mask: True where label is valid (not NaN)
        valid_mask   = ~torch.isnan(targets)           # (B, 12)
        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0.0  # NaN -> 0 in safe copy, masked out anyway

        # Gold upweighting: gold_weight for gold studies, 1.0 for pseudo
        sample_weights = torch.where(
            is_gold.bool().unsqueeze(1).expand_as(logits),
            torch.full_like(logits, self.gold_weight),
            torch.ones_like(logits),
        )  # (B, 12)

        # Optional per-label positive class weight
        if self.pos_weight is not None:
            pos_w = self.pos_weight.to(logits.device)
            label_weights = torch.where(
                safe_targets == 1.0,
                pos_w.unsqueeze(0).expand_as(logits),
                torch.ones_like(logits),
            )
            sample_weights = sample_weights * label_weights

        # Raw BCE (no reduction) then apply mask and sample weights
        bce      = F.binary_cross_entropy_with_logits(
            logits, safe_targets, reduction="none"
        )  # (B, 12)
        weighted = bce * sample_weights * valid_mask.float()  # zero out NaN cells

        n_valid = valid_mask.float().sum()
        if n_valid == 0:
            # Edge case: entire batch has no valid labels (unlikely but safe)
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        if self.reduction == "mean":
            return weighted.sum() / n_valid
        return weighted.sum()


class PlattScaler(nn.Module):
    """Temperature scaling for post-training calibration on gold studies.

    Uses a single shared temperature parameter T (not per-label) appropriate
    for the available calibration set size (n=58 gold studies).

    CRITICAL: Must be fitted on Out-of-Fold (OOF) validation logits collected
    during training, NOT on in-sample training predictions. train_fold()
    returns oof_gold_logits and oof_gold_targets for this purpose.

    Expected T > 1.0 for MIL models (typically overconfident).
    T < 1.0 is rare and suggests underconfidence.

    Args:
        n_classes: Output labels (12).
        init_temperature: Initial temperature (1.0 = no scaling).
    """

    def __init__(
        self,
        n_classes: int = 12,
        init_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature.

        Args:
            logits: (N, 12) raw logits.

        Returns:
            (N, 12) calibrated logits (still raw -- apply sigmoid after).
        """
        return logits / self.temperature.clamp(min=0.01)

    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        n_iters: int = 100,
        lr: float = 0.01,
    ) -> float:
        """Fit temperature using L-BFGS on OOF gold validation logits.

        Args:
            logits: (N_gold, 12) OOF validation logits (NOT in-sample).
            targets: (N_gold, 12) gold labels -- NaN should be filled to 0 first.
            n_iters: Max LBFGS iterations.
            lr: LBFGS learning rate.

        Returns:
            Fitted temperature scalar value.
        """
        self.train()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=n_iters)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                self.forward(logits), targets
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        t = self.temperature.item()
        print(
            f"   OOF Platt temperature: {t:.4f} "
            f"({'overconfident -> scaling down' if t > 1.0 else 'underconfident -> scaling up'})"
        )
        return t
