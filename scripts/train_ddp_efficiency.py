"""scripts/train_ddp_efficiency.py

Dual-GPU DistributedDataParallel (DDP) Training Engine for RSNA Efficiency Track.
Trains DINOv2-Small SlotHead across Kaggle Dual Tesla T4 GPUs with 3-stage progressive
unfreezing, LLRD learning rate schedule, and Top-3 checkpoint averaging.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import gc
import json
import logging
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence
import weakref

# Ensure project root is first on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure local user site packages can be discovered as fallback if not present (e.g. for pyarrow, matplotlib)
user_site = Path.home() / f".local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
if user_site.is_dir() and str(user_site) not in sys.path:
    sys.path.append(str(user_site))

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.datasets.efficiency_pipeline import DEFAULT_CROP_MM
from src.models.dinov2_slothead import DINOv2SlotHead
from src.training.losses import MaskedBCEWithLogitsLoss
from src.training.metrics import compute_competition_metrics, load_label_columns
from src.training.train_efficiency import (
    EfficiencyStudyDataset,
    build_differential_optimizer,
    load_pos_weights,
    seed_everything,
)
from src.utils.tensorboard import NoOpLogger, TensorBoardLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def lr_for(family: str, step: int, spe: int, epochs: int) -> float:
    """Compute explicit learning rate for parameter family at the given training step.

    Families:
      - 'head': Queries, classification head, slot embeddings.
      - 'bb_top': Blocks 8-11 + final norm.
      - 'bb_mid': Blocks 4-7.
      - 'bb_bot': Blocks 0-3, patch_embed, pos_embed, cls_token.
    """
    ep = step / max(1, spe)

    if ep < 2.0:
        # Stage 1: Head warmup and decay, backbone frozen
        if family != "head":
            return 0.0
        return (1e-4 + 9e-4 * ep) if ep < 1.0 else (1e-3 - 5e-4 * (ep - 1.0))

    if ep < 4.0:
        # Stage 2: Top blocks + Head
        p = (ep - 2.0) / 2.0
        if family == "head":
            return 5e-4 - 2e-4 * p
        if family == "bb_top":
            return 1e-5
        return 0.0

    # Stage 3: All blocks unfrozen, cosine decay to zero with gentle 100-step ramp
    ep3 = ep - 4.0
    p = min(1.0, ep3 / max(1.0, float(epochs - 4)))
    step_in_stage3 = step - 4 * spe
    ramp = min(1.0, max(0.05, float(step_in_stage3) / 100.0))

    peaks = {
        "bb_bot": 5e-6,
        "bb_mid": 1e-5,
        "bb_top": 2e-5,
        "head": 2e-4,
    }
    peak = peaks.get(family, 2e-4)
    base = peak * 0.5 * (1.0 + math.cos(math.pi * p))
    return (base * ramp) if ("bb" in family and step_in_stage3 < 100) else base


def apply_unfreezing_stage(raw_model: DINOv2SlotHead, stage: int) -> None:
    """Set parameter requires_grad flags according to unfreezing stage."""
    if stage == 1:
        for p in raw_model.backbone.parameters():
            p.requires_grad = False
    elif stage == 2:
        for p in raw_model.backbone.parameters():
            p.requires_grad = False
        for p in raw_model.backbone.blocks[8:].parameters():
            p.requires_grad = True
        for p in raw_model.backbone.norm.parameters():
            p.requires_grad = True
    elif stage == 3:
        for p in raw_model.backbone.parameters():
            p.requires_grad = True


def rewrap_ddp(
    raw_model: DINOv2SlotHead,
    device: torch.device,
    local_rank: int,
) -> DDP:
    """Instantiate DDP wrapper with find_unused_parameters=False."""
    if device.type == "cuda":
        return DDP(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    return DDP(raw_model, find_unused_parameters=False)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DistributedDataParallel or DataParallel wrapper to access base module."""
    return model.module if isinstance(model, (DDP, nn.DataParallel)) else model


def verify_cache(
    cache_dir: str | Path,
    uids: Sequence[str],
    target_size: int = 336,
    crop_mm: float = DEFAULT_CROP_MM,
) -> tuple[list[str], list[str]]:
    """Verify cache manifest provenance and check coverage before spawning DDP.

    Returns:
        (valid_uids: list[str], empty_uids: list[str])
    """
    d = Path(cache_dir)
    manifest_p = d / "cache_manifest.json"
    if not manifest_p.is_file():
        raise FileNotFoundError(f"Missing cache manifest at {manifest_p}. Run preprocess_volumes.py first.")

    with open(manifest_p, "r", encoding="utf-8") as f:
        m = json.load(f)

    if (
        m.get("n_slices_cached") != 5
        or m.get("target_size") != target_size
        or abs(float(m.get("crop_mm", 0.0)) - crop_mm) > 1e-4
    ):
        raise RuntimeError(f"Cache provenance mismatch: {m}")

    if m.get("in_progress", False):
        raise RuntimeError(f"Cache build at {d} is incomplete or was interrupted (in_progress=True). Re-run preprocess_volumes.py.")

    failed_set = set(m.get("failed_uids", []))
    empty_uids = list(m.get("empty_uids", []))
    max_allowed_failures = max(1, int(0.005 * len(uids)))
    if len(failed_set) > max_allowed_failures:
        raise RuntimeError(f"Cache failure rate ({len(failed_set)}) exceeds allowed 0.5% threshold ({max_allowed_failures})")

    valid_uids = [u for u in uids if u not in failed_set]
    missing = [u for u in valid_uids if not (d / f"{u}.pt").is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} studies missing from cache: e.g. {missing[:3]}")

    logger.info("Cache verification passed: %d valid studies (excluded %d failures, %d empty)", len(valid_uids), len(failed_set), len(empty_uids))
    return valid_uids, empty_uids


def gather_unique_records(
    records: list[dict[str, Any]],
    world_size: int,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    """Gather evaluation records across all ranks and deduplicate padded sampler entries."""
    gathered: list[list[dict[str, Any]] | None] = [None] * world_size
    dist.all_gather_object(gathered, records)
    unique: dict[str, dict[str, Any]] = {}
    for rank_recs in gathered:
        if rank_recs is None:
            continue
        for r in rank_recs:
            sid = str(r.get("study_id") or r.get("uid") or "")
            if sid and sid not in unique:
                unique[sid] = r
    return list(unique.values())


def evaluate_dataset_collective(
    ddp_model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    world_size: int,
    label_cols: list[str],
    is_rank0: bool,
    use_amp: bool = True,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Run rank-sharded validation and compute macro AUC on rank 0."""
    ddp_model.eval()
    local_records = []
    with torch.no_grad():
        for batch in val_loader:
            v_imgs = batch["image"].to(device, non_blocking=True).float()
            v_masks = batch["presence_mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                v_logits = ddp_model(v_imgs, v_masks)
            logits_np = v_logits.cpu().numpy()
            for i, uid in enumerate(batch["study_id"]):
                local_records.append({
                    "study_id": uid,
                    "logits": logits_np[i].tolist(),
                    "targets": batch["labels"][i].numpy().tolist(),
                    "is_gold": int(batch["is_gold"][i].item()),
                })

    all_val_records = gather_unique_records(local_records, world_size, device)

    score_11, score_12 = 0.0, 0.0
    if is_rank0:
        val_logits = np.array([r["logits"] for r in all_val_records])
        val_targets = np.array([r["targets"] for r in all_val_records])
        val_is_gold = np.array([r["is_gold"] for r in all_val_records])

        metrics = compute_competition_metrics(
            logits=val_logits,
            targets=val_targets,
            gold_mask=val_is_gold,
            label_cols=label_cols,
        )
        score_11 = metrics.macro_auc_11
        score_12 = metrics.macro_auc_12

    return score_11, score_12, all_val_records


def train_ddp_worker(
    local_rank: int,
    world_size: int,
    args: argparse.Namespace,
) -> None:
    """Single-process DDP worker executed per GPU."""
    seed_everything(args.seed + local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_rank0 = local_rank == 0

    backend = getattr(args, "backend", None) or ("nccl" if device.type == "cuda" else "gloo")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            rank=local_rank,
            world_size=world_size,
            timeout=datetime.timedelta(minutes=30),
        )

    # Telemetry logger on Rank 0
    tb_logger = NoOpLogger()
    if is_rank0 and args.tb_dir:
        try:
            tb_logger = TensorBoardLogger(args.tb_dir)
        except Exception as err:
            logger.warning("TensorBoard unavailable (%s); falling back to NoOpLogger.", err)
            tb_logger = NoOpLogger()

    # Ensure output weight directory exists before rank 0 saves
    if is_rank0:
        Path(args.weights_dir).mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    splits_p = Path(args.splits_path)
    splits_df = pd.read_parquet(splits_p) if splits_p.suffix == ".parquet" else pd.read_csv(splits_p)
    labels_p = Path(args.labels_path)
    labels_df = pd.read_parquet(labels_p) if labels_p.suffix == ".parquet" else pd.read_csv(labels_p)
    label_cols = load_label_columns(args.config_path)

    merged = splits_df.merge(labels_df, on="StudyInstanceUID", how="inner")

    # Filter out failed UIDs from all data
    if getattr(args, "valid_uids", None) is not None:
        merged = merged[merged["StudyInstanceUID"].astype(str).isin(args.valid_uids)]

    fold_col = "fold_id" if "fold_id" in merged.columns else "fold"
    train_df = merged[merged[fold_col] != args.fold].reset_index(drop=True)
    val_df = merged[merged[fold_col] == args.fold].reset_index(drop=True)

    # Drop empty studies from training only, preserving true distribution in validation
    if getattr(args, "empty_uids", None) is not None:
        train_df = train_df[~train_df["StudyInstanceUID"].astype(str).isin(args.empty_uids)].reset_index(drop=True)

    def _resolve_study_dirs(df_uids: Sequence[Any]) -> list[Path]:
        if args.dicom_root:
            return [Path(args.dicom_root) / str(u) for u in df_uids]
        if args.cache_dir:
            return [Path(args.cache_dir) / str(u) for u in df_uids]
        return [Path(str(u)) for u in df_uids]

    train_dataset = EfficiencyStudyDataset(
        study_uids=train_df["StudyInstanceUID"].tolist(),
        study_dirs=_resolve_study_dirs(train_df["StudyInstanceUID"]),
        labels=train_df[label_cols].values,
        is_gold=train_df["is_gold"].values if "is_gold" in train_df else None,
        series_csv_path=args.series_csv,
        cached_tensor_dir=args.cache_dir,
        target_size=args.img_size,
        crop_mm=args.crop_mm,
        is_train=True,
        require_cache=args.require_cache,
    )
    val_dataset = EfficiencyStudyDataset(
        study_uids=val_df["StudyInstanceUID"].tolist(),
        study_dirs=_resolve_study_dirs(val_df["StudyInstanceUID"]),
        labels=val_df[label_cols].values,
        is_gold=val_df["is_gold"].values if "is_gold" in val_df else None,
        series_csv_path=args.series_csv,
        cached_tensor_dir=args.cache_dir,
        target_size=args.img_size,
        crop_mm=args.crop_mm,
        is_train=False,
        require_cache=args.require_cache,
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True, seed=args.seed)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)

    pos_priors = train_df[label_cols].mean().values if label_cols else None

    raw_model = DINOv2SlotHead(
        pretrained=False,
        weights_path=args.weights_path,
        img_size=args.img_size,
        attn_dropout_p=0.2,
        pre_cls_dropout_p=0.3,
        init_bias_priors=pos_priors,
        slot_dropout_p=args.slot_dropout_p,
    ).to(device)

    optimizer = build_differential_optimizer(
        model=raw_model,
        lr_bottom=5e-6,
        lr_mid=1e-5,
        lr_top=2e-5,
        lr_head=1e-3,
        weight_decay=args.weight_decay,
    )

    raw_pos_weights = load_pos_weights(args.config_path)
    pos_weights_tensor = torch.as_tensor(raw_pos_weights, dtype=torch.float32, device=device)
    criterion = MaskedBCEWithLogitsLoss(gold_weight=args.gold_weight, pos_weight=pos_weights_tensor, continuous_pos_weight=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.use_amp and device.type == "cuda"))

    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum_steps))
    global_step = 0
    current_stage = 0
    ddp_model = None
    top_checkpoints: list[tuple[float, int, Path]] = []
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        stage = 1 if epoch <= 2 else (2 if epoch <= 4 else 3)
        if stage != current_stage:
            current_stage = stage
            apply_unfreezing_stage(raw_model, stage)

            # Verifiable caller teardown with weakref assertion
            ref = weakref.ref(ddp_model) if ddp_model is not None else None
            ddp_model = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if ref is not None and ref() is not None:
                logger.warning("Stale DDP wrapper reference still lingering!")
            if dist.is_initialized():
                dist.barrier()

            ddp_model = rewrap_ddp(raw_model, device, local_rank)
            if is_rank0:
                logger.info("Transitioned to Stage %d at Epoch %d", stage, epoch)

        ddp_model.train()
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum = 0.0
        step_count = 0

        for step_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True).float()
            masks = batch["presence_mask"].to(device, non_blocking=True)
            targets = batch["labels"].to(device, non_blocking=True)
            is_gold = batch["is_gold"].to(device, non_blocking=True)

            is_accumulating = ((step_idx + 1) % args.grad_accum_steps != 0) and (step_idx + 1 < len(train_loader))
            sync_context = ddp_model.no_sync() if is_accumulating else contextlib.nullcontext()

            # Set explicit learning rates per param group before optimizer step
            for g in optimizer.param_groups:
                family = g["name"].rsplit("_", 1)[0]
                g["lr"] = lr_for(family, global_step, steps_per_epoch, args.epochs)

            with sync_context:
                with torch.amp.autocast("cuda", enabled=(args.use_amp and device.type == "cuda")):
                    logits = ddp_model(images, masks)
                    raw_loss = criterion(logits, targets, is_gold=is_gold)
                    loss = raw_loss / args.grad_accum_steps

                scaler.scale(loss).backward()
                epoch_loss_sum += raw_loss.item()
                step_count += 1

            if not is_accumulating:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        # Clear references to training graph before validation
        logits = loss = raw_loss = None

        # Distributed validation across all ranks
        score_11, score_12, _ = evaluate_dataset_collective(
            ddp_model=ddp_model,
            val_loader=val_loader,
            device=device,
            world_size=world_size,
            label_cols=label_cols,
            is_rank0=is_rank0,
            use_amp=args.use_amp,
        )

        if is_rank0:
            avg_train_loss = epoch_loss_sum / max(1, step_count)
            logger.info("Epoch %d Validation Macro AUC 11: %.4f (Macro AUC 12: %.4f)", epoch, score_11, score_12)
            tb_logger.log_scalar("val/macro_auc_11", score_11, epoch)
            tb_logger.log_scalar("val/macro_auc_12", score_12, epoch)
            tb_logger.log_scalar("train/loss", avg_train_loss, epoch)

            history.append({
                "epoch": epoch,
                "train_loss": float(avg_train_loss),
                "val_loss": float(score_11),
                "macro_auc_11": float(score_11),
                "macro_auc_12": float(score_12),
                "score": float(score_11),
            })

            ckpt_path = Path(args.weights_dir) / f"dinov2_slothead_epoch_{epoch:02d}.pth"
            torch.save({"model_state_dict": raw_model.state_dict(), "epoch": epoch, "score": score_11}, ckpt_path)

            # Top-3 tracking strictly in Stage 3 (epoch >= 5 when total epochs >= 5, else epoch >= 1)
            stage3_start = 5 if args.epochs >= 5 else 1
            if epoch >= stage3_start and not math.isnan(score_11):
                top_checkpoints.append((score_11, epoch, ckpt_path))
                top_checkpoints.sort(key=lambda x: x[0], reverse=True)
                if len(top_checkpoints) > 3:
                    _, _, old_p = top_checkpoints.pop()
                    if old_p.is_file():
                        old_p.unlink()

    # Top-3 Checkpoint Averaging & Collective Validation
    averaged_success = False
    top3_path = Path(args.weights_dir) / "dinov2_slothead_top3.pth"
    best_single_path = Path(args.weights_dir) / "dinov2_slothead_best_single.pth"
    ens_score: float | None = None

    if is_rank0 and len(top_checkpoints) >= 1:
        best_single_ckpt = torch.load(top_checkpoints[0][2], map_location="cpu")
        torch.save(best_single_ckpt["model_state_dict"], best_single_path)
        logger.info("Saved best single checkpoint (Epoch %d, Score %.4f) to %s", top_checkpoints[0][1], top_checkpoints[0][0], best_single_path)

        if len(top_checkpoints) >= 2:
            top_paths = [p for _, _, p in top_checkpoints[:3]]
            logger.info("Averaging Stage-3 checkpoints: %s", [p.name for p in top_paths])
            state_dicts = [torch.load(p, map_location="cpu")["model_state_dict"] for p in top_paths]
            avg_state = {}
            for k in state_dicts[0]:
                avg_state[k] = torch.stack([sd[k].float() for sd in state_dicts]).mean(dim=0)
            torch.save(avg_state, top3_path)
            averaged_success = True

    # Broadcast averaging status to all ranks
    avg_flag = [averaged_success]
    dist.broadcast_object_list(avg_flag, src=0)

    recommended_path = best_single_path if best_single_path.is_file() else None
    if avg_flag[0]:
        # Collective evaluation of the averaged weights
        raw_model.load_state_dict(torch.load(top3_path, map_location=device))
        ens_score, _, _ = evaluate_dataset_collective(
            ddp_model=ddp_model,
            val_loader=val_loader,
            device=device,
            world_size=world_size,
            label_cols=label_cols,
            is_rank0=is_rank0,
            use_amp=args.use_amp,
        )
        if is_rank0:
            best_single_score = top_checkpoints[0][0] if top_checkpoints else 0.0
            logger.info("Ensemble Top-3 AUC 11: %.4f vs Best Single Epoch AUC 11: %.4f", ens_score, best_single_score)
            if ens_score >= best_single_score:
                recommended_path = top3_path
            else:
                recommended_path = best_single_path

    if is_rank0:
        chosen_auc_11 = ens_score if (averaged_success and ens_score is not None) else (top_checkpoints[0][0] if top_checkpoints else None)
        results_summary = {
            "history": history,
            "averaged_macro_auc_11": float(chosen_auc_11) if chosen_auc_11 is not None else None,
            "recommended_weights": str(recommended_path) if recommended_path and recommended_path.is_file() else None,
            "recommended_weights_path": str(recommended_path) if recommended_path and recommended_path.is_file() else None,
            "top3_averaged": str(top3_path) if averaged_success else None,
            "best_single": str(best_single_path) if best_single_path.is_file() else None,
            "best_single_score": float(top_checkpoints[0][0]) if top_checkpoints else None,
        }
        with open(Path(args.weights_dir) / "ddp_results.json", "w", encoding="utf-8") as f:
            json.dump(results_summary, f, indent=2)

    if is_rank0:
        tb_logger.close()
    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="DDP Training Runner for RSNA Efficiency Track")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=336)
    parser.add_argument("--crop-mm", type=float, default=DEFAULT_CROP_MM)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gold-weight", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-cache", action="store_true", default=False)
    parser.add_argument("--cache-dir", type=str, default="/kaggle/tmp/rsna_cache")
    parser.add_argument("--weights-path", type=str, default=None)
    parser.add_argument("--allow-random-init", action="store_true", default=False)
    parser.add_argument("--slot-dropout-p", type=float, default=0.0)
    parser.add_argument("--weights-dir", type=str, default="outputs/weights")
    parser.add_argument("--tb-dir", type=str, default="outputs/tensorboard")
    parser.add_argument("--dicom-root", type=str, default=None)
    parser.add_argument("--series-csv", type=str, default=None)
    parser.add_argument("--splits-path", type=str, required=True)
    parser.add_argument("--labels-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, default="config.yaml")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    # Pre-spawn fail-closed weights guard
    if not args.weights_path and not args.allow_random_init:
        raise ValueError("Must supply --weights-path or explicitly pass --allow-random-init.")

    # Pre-spawn 100% cache verification
    if args.require_cache and args.cache_dir:
        splits_df = pd.read_csv(args.splits_path)
        uids = splits_df["StudyInstanceUID"].astype(str).tolist()
        valid_uids, empty_uids = verify_cache(args.cache_dir, uids, target_size=args.img_size, crop_mm=args.crop_mm)
        args.valid_uids = frozenset(valid_uids)
        args.empty_uids = frozenset(empty_uids)

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if world_size > 1:
        torch.multiprocessing.spawn(train_ddp_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train_ddp_worker(local_rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()
