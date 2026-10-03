"""tests/test_train_efficiency.py

Unit and integration test suite for Issue 05: Single-GPU Training Recipe.
Verifies:
  1. Differential AdamW optimizer parameter groups & learning rates.
  2. OneCycleLR schedule cycle.
  3. Top-3 checkpoint arithmetic state dict averaging.
  4. Checkpoint eviction (guaranteed deletion of worst checkpoints without leaks).
  5. Deterministic single-batch overfitting test verifying strict loss decrease.
  6. Soft label routing preserving explicit labels and gold studies.
  7. Dual float & bool presence_mask model acceptance contract.
  8. Pretrained backbone weights loading & pos_embed resampling contract.
  9. Loss function NaN-masking finite loss and gradient contract.
  10. Competition metrics soft-target evaluation contract.
  11. Integration smoke test: 2 epochs on 20 studies with accum in {1, 3}, verifying checkpoint saving, averaged model validation, and finite loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import timm
import torch
import torch.nn as nn

from src.datasets.efficiency_pipeline import NUM_SLOTS
from src.models.dinov2_slothead import DINOv2SlotHead
from src.training.losses import MaskedBCEWithLogitsLoss
from src.training.metrics import compute_competition_metrics
from src.training.train_efficiency import (
    _register_checkpoint,
    average_top_checkpoints,
    build_differential_optimizer,
    load_labels_and_splits,
    load_pos_weights,
    train_efficiency_fold,
)

LABEL_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"
]


def test_differential_optimizer_parameter_groups():
    """Verify differential optimizer splits parameters into 4 groups with correct LRs and decays."""
    model = DINOv2SlotHead(pretrained=False, img_size=112)
    optimizer = build_differential_optimizer(
        model=model,
        lr_backbone=3e-5,
        lr_head=1e-3,
        weight_decay=0.02,
    )

    assert len(optimizer.param_groups) == 4, f"Expected 4 groups, found {len(optimizer.param_groups)}"

    # Group 0: Backbone decay
    assert optimizer.param_groups[0]["lr"] == 3e-5
    assert optimizer.param_groups[0]["weight_decay"] == 0.02
    assert len(optimizer.param_groups[0]["params"]) > 0

    # Group 1: Backbone no-decay
    assert optimizer.param_groups[1]["lr"] == 3e-5
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    assert len(optimizer.param_groups[1]["params"]) > 0

    # Group 2: Head decay (cross_attn weights, cls_w)
    assert optimizer.param_groups[2]["lr"] == 1e-3
    assert optimizer.param_groups[2]["weight_decay"] == 0.02
    assert len(optimizer.param_groups[2]["params"]) > 0

    # Group 3: Head no-decay (biases, norms, slot_embed, finding_queries)
    assert optimizer.param_groups[3]["lr"] == 1e-3
    assert optimizer.param_groups[3]["weight_decay"] == 0.0
    assert len(optimizer.param_groups[3]["params"]) > 0

    # Total parameters accounted for exactly
    total_opt_params = sum(len(g["params"]) for g in optimizer.param_groups)
    total_model_params = len([p for p in model.parameters() if p.requires_grad])
    assert total_opt_params == total_model_params


def test_onecycle_lr_schedule_cycle():
    """Verify OneCycleLR warms up to peak learning rates and decays smoothly."""
    model = nn.Linear(10, 2)
    opt = torch.optim.AdamW([
        {"params": [model.weight], "lr": 3e-5},
        {"params": [model.bias], "lr": 1e-3},
    ])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=[3e-5, 1e-3],
        total_steps=100,
        pct_start=0.15,
    )

    initial_lr_bb = opt.param_groups[0]["lr"]
    initial_lr_hd = opt.param_groups[1]["lr"]

    assert np.isclose(initial_lr_bb, 3e-5 / 25.0)
    assert np.isclose(initial_lr_hd, 1e-3 / 25.0)

    # Step to 15% (peak)
    for _ in range(15):
        opt.step()
        scheduler.step()

    assert np.isclose(opt.param_groups[0]["lr"], 3e-5, rtol=1e-2)
    assert np.isclose(opt.param_groups[1]["lr"], 1e-3, rtol=1e-2)

    # Step to end (decay)
    for _ in range(85):
        opt.step()
        scheduler.step()

    assert opt.param_groups[0]["lr"] < 3e-5 / 100.0
    assert opt.param_groups[1]["lr"] < 1e-3 / 100.0


def test_average_top_checkpoints(tmp_path: Path):
    """Verify checkpoint averaging computes exact parameter-wise arithmetic mean."""
    ckpt1 = tmp_path / "ckpt1.pth"
    ckpt2 = tmp_path / "ckpt2.pth"
    ckpt3 = tmp_path / "ckpt3.pth"
    out_file = tmp_path / "averaged.pth"

    w1 = {"weight": torch.tensor([1.0, 2.0]), "step": torch.tensor(10)}
    w2 = {"weight": torch.tensor([3.0, 4.0]), "step": torch.tensor(20)}
    w3 = {"weight": torch.tensor([5.0, 6.0]), "step": torch.tensor(30)}

    torch.save({"model_state_dict": w1}, ckpt1)
    torch.save({"model_state_dict": w2}, ckpt2)
    torch.save({"model_state_dict": w3}, ckpt3)

    average_top_checkpoints([ckpt1, ckpt2, ckpt3], out_file)
    assert out_file.is_file()

    avg_state = torch.load(out_file, weights_only=True)
    expected_w = torch.tensor([3.0, 4.0])
    assert torch.allclose(avg_state["weight"], expected_w)
    assert avg_state["step"].item() == 10


def test_register_checkpoint_evicts_worst_even_if_new(tmp_path: Path):
    """Verify _register_checkpoint evicts and deletes whichever file ranks below top-k."""
    top: list[tuple[float, int, Path]] = []
    files = []
    for ep, score in enumerate([0.9, 0.8, 0.7, 0.1], start=1):
        p = tmp_path / f"c{ep}.pth"
        p.write_bytes(b"placeholder")
        files.append(p)
        _register_checkpoint(top, score, ep, p, k=3)

    assert len(top) == 3
    # Checkpoint 4 had lowest score (0.1), must be deleted from disk immediately
    assert not files[3].exists()
    assert all(f.exists() for f in files[:3])


def test_overfit_single_batch_loss_decreases():
    """Deterministic verification of the 'loss decrease' acceptance criterion."""
    torch.manual_seed(42)
    model = DINOv2SlotHead(pretrained=False, img_size=112, slot_dropout_p=0.0)
    pos_w = torch.tensor(load_pos_weights(), dtype=torch.float32)
    criterion = MaskedBCEWithLogitsLoss(gold_weight=5.0, pos_weight=pos_w, continuous_pos_weight=True)
    opt = build_differential_optimizer(model, lr_backbone=1e-4, lr_head=1e-3)

    x = torch.randn(4, NUM_SLOTS, 3, 112, 112)
    m = torch.ones(4, NUM_SLOTS)
    y = (torch.rand(4, 12) > 0.5).float()
    g = torch.zeros(4, dtype=torch.long)

    model.train()
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = criterion(model(x, m), y, g)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # Loss must strictly decrease by at least 20%
    assert losses[-1] < 0.80 * losses[0], f"Loss failed to decrease: start {losses[0]}, end {losses[-1]}"


def test_load_labels_and_splits_synovitis_soft_routing(tmp_path: Path):
    """Verify synovitis_soft is authoritative for non-gold rows; gold rows are never modified."""
    splits_csv = tmp_path / "splits.csv"
    labels_csv = tmp_path / "labels.csv"

    # 4 studies: 2 gold, 2 pseudo
    splits_data = {
        "StudyInstanceUID": ["s1", "s2", "s3", "s4"],
        "fold_id": [0, 0, 1, 1],
        "is_gold": [1, 0, 1, 0],
    }
    pd.DataFrame(splits_data).to_csv(splits_csv, index=False)

    labels_data = {
        "StudyInstanceUID": ["s1", "s2", "s3", "s4"],
        "Synovitis": [1.0, 0.0, 0.0, 1.0],
        "synovitis_soft": [float("nan"), 0.63, float("nan"), 0.22],
    }
    for col in LABEL_COLS:
        if col != "Synovitis":
            labels_data[col] = [0.0, 1.0, float("nan"), 0.0]
    pd.DataFrame(labels_data).to_csv(labels_csv, index=False)

    train_df, val_df, _ = load_labels_and_splits(
        labels_path=labels_csv,
        splits_path=splits_csv,
        fold_id=0,
        label_cols=LABEL_COLS,
    )

    # In val_df (fold 0): s1 is gold -> keeps 1.0; s2 is non-gold -> receives 0.63
    s1_row = val_df[val_df["StudyInstanceUID"] == "s1"].iloc[0]
    s2_row = val_df[val_df["StudyInstanceUID"] == "s2"].iloc[0]
    assert s1_row["Synovitis"] == 1.0
    assert s2_row["Synovitis"] == 0.63

    # In train_df (fold 1): s3 is gold -> keeps 0.0; s4 is non-gold -> receives 0.22
    s3_row = train_df[train_df["StudyInstanceUID"] == "s3"].iloc[0]
    s4_row = train_df[train_df["StudyInstanceUID"] == "s4"].iloc[0]
    assert s3_row["Synovitis"] == 0.0
    assert s4_row["Synovitis"] == 0.22


def test_model_accepts_float_and_bool_presence_masks():
    """Verify DINOv2SlotHead gathers slots identically under float32 and boolean presence masks."""
    torch.manual_seed(0)
    model = DINOv2SlotHead(pretrained=False, img_size=112, slot_dropout_p=0.0).eval()
    x = torch.randn(2, NUM_SLOTS, 3, 112, 112)
    m_f = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 0, 1, 0, 1, 1]], dtype=torch.float32)
    with torch.no_grad():
        out_f = model(x, m_f)
        out_b = model(x, m_f.bool())
    assert out_f.shape == (2, 12)
    assert torch.allclose(out_f, out_b, atol=1e-5)


def test_backbone_weights_path_loads_and_resamples_pos_embed(tmp_path: Path):
    """Verify offline 518px checkpoint loads into a 112px model with timm pos_embed resampling."""
    ref = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0, img_size=518)
    with torch.no_grad():
        for p in ref.parameters():
            p.fill_(0.01)
    ckpt = tmp_path / "dinov2_s_518.pth"
    torch.save(ref.state_dict(), ckpt)

    model = DINOv2SlotHead(pretrained=False, weights_path=str(ckpt), img_size=112)
    bb = model.backbone
    assert bb.pos_embed.shape == (1, (112 // 14) ** 2 + 1, 384)
    for name in ("pos_embed", "cls_token"):
        t = getattr(bb, name)
        assert torch.allclose(t, torch.full_like(t, 0.01), atol=1e-4), name
    w = bb.blocks[0].attn.qkv.weight
    assert torch.allclose(w, torch.full_like(w, 0.01), atol=1e-6)


def test_masked_loss_nan_targets_give_finite_loss_and_grads():
    """Verify MaskedBCEWithLogitsLoss masks NaNs before multiplication, preventing grad leaks."""
    logits = torch.randn(4, 12, requires_grad=True)
    y = torch.rand(4, 12).round()
    y[0, 0] = float("nan")
    y[:, 5] = float("nan")  # fully silent column
    g = torch.tensor([0, 1, 0, 0])
    crit = MaskedBCEWithLogitsLoss(gold_weight=5.0, pos_weight=torch.tensor(load_pos_weights()), continuous_pos_weight=True)
    loss = crit(logits, y, g)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0] == 0 and (logits.grad[:, 5] == 0).all()


def test_competition_metrics_accept_soft_targets():
    """Verify compute_competition_metrics evaluates continuous pseudo-labels without exceptions."""
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(40, 12, generator=g)
    targets = torch.rand(40, 12, generator=g)  # soft continuous targets in [0, 1]
    gold = torch.zeros(40, dtype=torch.bool)
    gold[:20] = True
    targets[:20, LABEL_COLS.index("Synovitis")] = (torch.rand(20, generator=g) > 0.5).float()
    m = compute_competition_metrics(logits=logits, targets=targets, gold_mask=gold, label_cols=LABEL_COLS)
    assert 0.0 <= float(m.macro_auc_11) <= 1.0


@pytest.mark.parametrize("accum", [1, 3])
def test_train_efficiency_2epoch_smoke_integration(tmp_path: Path, accum: int):
    """Integration smoke test: 2 epochs on 20 studies with grad_accum_steps in {1, 3}."""
    n_studies = 20
    uids = [f"study_{i:03d}" for i in range(n_studies)]

    # 1. Create temporary cached float16 tensors (6, 3, 112, 112) with metadata
    cache_dir = tmp_path / "tensor_cache"
    cache_dir.mkdir()
    for uid in uids:
        tensor = torch.randn(NUM_SLOTS, 3, 112, 112, dtype=torch.float16)
        presence = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
        meta = {"size": 112, "crop_mm": 140.0}
        torch.save({"image": tensor, "presence_mask": presence, "meta": meta}, cache_dir / f"{uid}.pt")

    # 2. Create splits CSV (16 train, 4 val; 2 gold studies in val)
    splits_csv = tmp_path / "splits.csv"
    fold_ids = [0 if i < 4 else 1 for i in range(n_studies)]
    is_gold = [1 if i in (0, 1, 4, 5) else 0 for i in range(n_studies)]
    splits_df = pd.DataFrame({
        "StudyInstanceUID": uids,
        "fold_id": fold_ids,
        "is_gold": is_gold,
    })
    splits_df.to_csv(splits_csv, index=False)

    # 3. Create labels parquet with varying values per label column
    labels_parquet = tmp_path / "labels_soft.parquet"
    labels_dict: dict[str, Any] = {"StudyInstanceUID": uids}
    for j, col in enumerate(LABEL_COLS):
        vals = [float((i + j) % 2 == 0 if (i + j) % 7 != 0 else float("nan")) for i in range(n_studies)]
        labels_dict[col] = vals
    labels_dict["synovitis_soft"] = [0.63 if i % 2 == 0 else 0.22 for i in range(n_studies)]
    pd.DataFrame(labels_dict).to_parquet(labels_parquet)

    weights_dir = tmp_path / f"weights_accum{accum}"
    mock_dirs = {uid: tmp_path / uid for uid in uids}

    # Run 2-epoch training integration
    results = train_efficiency_fold(
        fold_id=0,
        n_epochs=2,
        batch_size=4,
        grad_accum_steps=accum,
        lr_backbone=3e-5,
        lr_head=1e-3,
        img_size=112,
        allow_random_init=True,
        labels_path=labels_parquet,
        splits_path=splits_csv,
        cached_tensor_dir=cache_dir,
        weights_dir=weights_dir,
        tb_log_dir=None,
        device="cpu",
        use_amp=False,
        num_workers=0,
        study_dirs_map=mock_dirs,
    )

    # Verify return dictionary contracts
    assert "best_score" in results
    assert "best_epoch" in results
    assert len(results["top3_checkpoints"]) == 2
    assert Path(results["recommended_weights_path"]).is_file()
    assert "history" in results

    h = results["history"]
    assert len(h) == 2
    assert all(np.isfinite(x["train_loss"]) and np.isfinite(x["val_loss"]) for x in h)
    assert h[-1]["train_loss"] <= h[0]["train_loss"] * 1.05

    # Verify checkpoint directory strictly has <= 3 epoch files
    saved_ckpts = list(weights_dir.glob("checkpoint_epoch_*.pth"))
    assert len(saved_ckpts) <= 3

    # Verify recommended weights load cleanly
    rec_weights = torch.load(results["recommended_weights_path"], weights_only=True)
    assert "finding_queries" in rec_weights
    assert "slot_embed" in rec_weights
    assert "cls_w" in rec_weights
    assert "cls_b" in rec_weights
    assert rec_weights["cls_w"].shape == (12, 384)
