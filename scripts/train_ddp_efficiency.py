"""scripts/train_ddp_efficiency.py

Dual-GPU DistributedDataParallel (DDP) Training Engine for RSNA Efficiency Track.
Trains DINOv2-Small SlotHead across Kaggle Dual Tesla T4 GPUs with 4-stage progressive
unfreezing, 4-tier LLRD learning rate schedule, continuous soft weak supervision,
Stage 4-restricted Top-3 checkpoint averaging, and true validation loss telemetry.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

from src.datasets.efficiency_pipeline import DEFAULT_CROP_MM, verify_cache
from src.models.dinov2_slothead import DINOv2SlotHead
from src.training.ddp import advance_stage, rewrap_ddp, set_stage_trainable
apply_unfreezing_stage = set_stage_trainable
from src.training.losses import MaskedBCEWithLogitsLoss
from src.training.metrics import compute_competition_metrics, load_label_columns
from src.training.train_efficiency import (
    STAGE4_LR_PEAKS,
    EfficiencyStudyDataset,
    build_differential_optimizer,
    compute_collective_val_loss,
    impute_soft_synovitis,
    load_pos_weights,
    seed_everything,
    stage_for_epoch,
    stage4_start_for,
)
from src.utils.tensorboard import NoOpLogger, TensorBoardLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def lr_for(family: str, step: int, spe: int, epochs: int) -> float:
    """Compute explicit learning rate for parameter family at the given training step.

    Applies a 100-step linear warmup ramp strictly to newly unfrozen layers at each stage boundary.
    Parameterizes stage boundaries to eliminate duplicate code across 14-epoch and 10-epoch runs.
    """
    e1, e2, e3 = (2, 4, 7) if epochs >= 14 else (2, 4, 6)
    ep = step / max(1, spe)
    ramp = lambda start: min(1.0, max(0.05, float(step - start * spe) / 100.0))

    if ep < e1:
        if family != "head":
            return 0.0
        return (1e-4 + 9e-4 * ep) if ep < 1.0 else (1e-3 - 5e-4 * (ep - 1.0))

    if ep < e2:
        if family == "head":
            return 5e-4 - 2e-4 * (ep - e1) / float(e2 - e1)
        return 1e-5 * ramp(e1) if family == "bb_top2" else 0.0

    if ep < e3:
        if family == "head":
            return 2e-4
        if family == "bb_top2":
            return 1.5e-5
        return 1e-5 * ramp(e2) if family == "bb_top1" else 0.0

    # Stage 4: Cosine decay across remaining epochs
    p = min(1.0, (ep - e3) / max(1.0, float(epochs - e3)))
    peak = STAGE4_LR_PEAKS.get(family, 2e-4)
    base = peak * 0.5 * (1.0 + math.cos(math.pi * p))
    return base * ramp(e3) if family in ("bb_mid", "bb_bot") else base


def unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DistributedDataParallel or DataParallel wrapper to access base module."""
    return model.module if isinstance(model, (DDP, nn.DataParallel)) else model


def gather_unique_records(
    local_records: list[dict[str, Any]],
    world_size: int,
) -> list[dict[str, Any]]:
    """Gather records from all ranks and deduplicate on study_id."""
    if world_size <= 1:
        seen = set()
        unique = []
        for r in local_records:
            sid = str(r.get("study_id") or r.get("uid") or "")
            if sid not in seen:
                seen.add(sid)
                unique.append(r)
        return unique

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_records)

    all_records = []
    seen = set()
    for rank_records in gathered:
        if rank_records is not None:
            for r in rank_records:
                sid = str(r.get("study_id") or r.get("uid") or "")
                if sid not in seen:
                    seen.add(sid)
                    all_records.append(r)
    return all_records


def evaluate_dataset_collective(
    ddp_model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    world_size: int,
    label_cols: list[str],
    is_rank0: bool,
    use_amp: bool = True,
    criterion: nn.Module | None = None,
) -> tuple[float, float, float, int, float, list[dict[str, Any]]]:
    """Run rank-sharded validation and compute macro AUC & true GPU-safe BCE loss on rank 0.

    Returns:
        tuple of (score_11, score_12, score_gold, gold_n_labels, val_loss, all_val_records)
    """
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

    all_val_records = gather_unique_records(local_records, world_size)

    score_11, score_12, score_gold, gold_n_labels, val_loss = 0.0, 0.0, float("nan"), 0, 0.0
    if is_rank0:
        val_logits_np = np.array([r["logits"] for r in all_val_records])
        val_targets_np = np.array([r["targets"] for r in all_val_records])
        val_is_gold_np = np.array([r["is_gold"] for r in all_val_records])

        metrics = compute_competition_metrics(
            logits=val_logits_np,
            targets=val_targets_np,
            gold_mask=val_is_gold_np,
            label_cols=label_cols,
        )
        score_11 = metrics.macro_auc_11
        score_12 = metrics.macro_auc_12

        # Evaluate Gold subset AUC cleanly without false 0.0 logging
        gold_mask_arr = (val_is_gold_np == 1)
        if gold_mask_arr.any():
            try:
                gold_metrics = compute_competition_metrics(
                    logits=val_logits_np[gold_mask_arr],
                    targets=val_targets_np[gold_mask_arr],
                    gold_mask=np.ones(gold_mask_arr.sum(), dtype=np.int64),
                    label_cols=label_cols,
                )
                score_gold = gold_metrics.macro_auc_11
                gold_n_labels = gold_metrics.n_valid_labels
            except Exception as e:
                logger.warning("Gold subset AUC undefined: %s", e)
                score_gold = float("nan")
                gold_n_labels = 0

        val_loss = compute_collective_val_loss(
            val_logits_np=val_logits_np,
            val_targets_np=val_targets_np,
            val_is_gold_np=val_is_gold_np,
            criterion=criterion,
            device=device,
        )

    return score_11, score_12, score_gold, gold_n_labels, val_loss, all_val_records


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
            timeout=datetime.timedelta(seconds=1800),
        )

    tb_logger = NoOpLogger()
    if is_rank0 and args.tb_dir:
        try:
            tb_logger = TensorBoardLogger(args.tb_dir)
        except Exception as err:
            logger.warning("TensorBoard unavailable (%s); falling back to NoOpLogger.", err)
            tb_logger = NoOpLogger()

    if is_rank0:
        Path(args.weights_dir).mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    splits_p = Path(args.splits_path)
    splits_df = pd.read_parquet(splits_p) if splits_p.suffix == ".parquet" else pd.read_csv(splits_p)
    labels_p = Path(args.labels_path)
    labels_df = pd.read_parquet(labels_p) if labels_p.suffix == ".parquet" else pd.read_csv(labels_p)
    label_cols = load_label_columns(args.config_path)
    if "is_gold" in labels_df.columns and "is_gold" in splits_df.columns:
        labels_df = labels_df.drop(columns=["is_gold"])
    merged = splits_df.merge(labels_df, on="StudyInstanceUID", how="inner")
    if "is_gold" not in merged.columns and "is_gold_x" in merged.columns:
        merged["is_gold"] = merged["is_gold_x"]

    if getattr(args, "valid_uids", None) is not None:
        merged = merged[merged["StudyInstanceUID"].astype(str).isin(args.valid_uids)]

    fold_col = "fold_id" if "fold_id" in merged.columns else "fold"
    train_df = merged[merged[fold_col] != args.fold].reset_index(drop=True)
    val_df = merged[merged[fold_col] == args.fold].reset_index(drop=True)

    if getattr(args, "empty_uids", None) is not None:
        train_df = train_df[~train_df["StudyInstanceUID"].astype(str).isin(args.empty_uids)].reset_index(drop=True)

    # Continuous soft weak supervision for Synovitis
    if args.soft_synovitis:
        n_imputed = impute_soft_synovitis(train_df, label_cols)
        if is_rank0:
            logger.info("Soft weak supervision active: imputed %d silent non-gold train studies", n_imputed)
    else:
        if is_rank0:
            logger.info("Soft weak supervision disabled (--no-soft-synovitis): silent rows left as NaN")

    # 2d. Soft-label premise preflight check
    labels_mat = train_df[label_cols].values
    frac_counts = ((labels_mat > 0) & (labels_mat < 1)).sum(axis=0)
    frac_series = pd.Series(frac_counts, index=label_cols)
    if is_rank0:
        logger.info("Fractional soft targets count per label:\n%s", frac_series)
    if args.soft_synovitis and frac_counts.sum() == 0:
        raise ValueError("Labels file has no fractional targets; soft supervision would be a no-op.")

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
        lr_top1=1.5e-5,
        lr_top2=2e-5,
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
    stage4_start = stage4_start_for(args.epochs)

    for epoch in range(1, args.epochs + 1):
        stage = stage_for_epoch(epoch, total_epochs=args.epochs)

        if stage != current_stage:
            current_stage = stage
            # Drop previous reference before advancing stage
            ddp_model = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            ddp_model = advance_stage(
                raw_model=raw_model,
                stage=stage,
                local_rank=local_rank,
            )
            if is_rank0:
                logger.info("Transitioned to Stage %d at Epoch %d", stage, epoch)

        ddp_model.train()
        if current_stage == 1:
            raw_model.backbone.eval()

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

        logits = loss = raw_loss = None

        try:
            score_11, score_12, score_gold, gold_n_labels, val_loss, _ = evaluate_dataset_collective(
                ddp_model=ddp_model,
                val_loader=val_loader,
                device=device,
                world_size=world_size,
                label_cols=label_cols,
                is_rank0=is_rank0,
                use_amp=args.use_amp,
                criterion=criterion,
            )
        except Exception as val_err:
            if is_rank0:
                logger.error("Epoch %d validation failed: %s", epoch, val_err)
            raise

        if is_rank0:
            avg_train_loss = epoch_loss_sum / max(1, step_count)
            logger.info("Epoch %d Val Macro AUC 11: %.4f (AUC 12: %.4f, Gold: %.4f [%d labels], Val Loss: %.4f, Train Loss: %.4f)",
                        epoch, score_11, score_12, score_gold, gold_n_labels, val_loss, avg_train_loss)
            tb_logger.log_scalar("val/macro_auc_11", score_11, epoch)
            tb_logger.log_scalar("val/macro_auc_12", score_12, epoch)
            tb_logger.log_scalar("val/loss", val_loss, epoch)
            tb_logger.log_scalar("train/loss", avg_train_loss, epoch)
            if not math.isnan(score_gold):
                tb_logger.log_scalar("val/macro_auc_gold", score_gold, epoch)
                tb_logger.log_scalar("val/gold_n_labels", gold_n_labels, epoch)

            history.append({
                "epoch": epoch,
                "train_loss": float(avg_train_loss),
                "val_loss": float(val_loss),
                "macro_auc_11": float(score_11),
                "macro_auc_12": float(score_12),
                "macro_auc_gold": float(score_gold) if not math.isnan(score_gold) else None,
                "score": float(score_11),
            })

            # Save and track checkpoints strictly in Stage 4 (zero intermediate disk writes in Stages 1-3)
            if epoch >= stage4_start and not math.isnan(score_11):
                ckpt_path = Path(args.weights_dir) / f"dinov2_slothead_epoch_{epoch:02d}.pth"
                torch.save({"model_state_dict": raw_model.state_dict(), "epoch": epoch, "score": score_11}, ckpt_path)
                top_checkpoints.append((score_11, epoch, ckpt_path))
                top_checkpoints.sort(key=lambda x: x[0], reverse=True)
                if len(top_checkpoints) > 3:
                    _, _, old_p = top_checkpoints.pop()
                    if old_p.is_file():
                        old_p.unlink()

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
            logger.info("Averaging Stage-4 checkpoints: %s", [p.name for p in top_paths])
            state_dicts = [torch.load(p, map_location="cpu")["model_state_dict"] for p in top_paths]
            avg_state = {}
            for k in state_dicts[0]:
                avg_state[k] = torch.stack([sd[k].float() for sd in state_dicts]).mean(dim=0)
            torch.save(avg_state, top3_path)
            averaged_success = True

    avg_flag = [averaged_success]
    dist.broadcast_object_list(avg_flag, src=0)

    recommended_path = best_single_path if best_single_path.is_file() else None
    if avg_flag[0]:
        raw_model.load_state_dict(torch.load(top3_path, map_location=device))
        ens_score, _, _, _, _, _ = evaluate_dataset_collective(
            ddp_model=ddp_model,
            val_loader=val_loader,
            device=device,
            world_size=world_size,
            label_cols=label_cols,
            is_rank0=is_rank0,
            use_amp=args.use_amp,
            criterion=criterion,
        )
        if is_rank0:
            best_single_score = top_checkpoints[0][0] if top_checkpoints else 0.0
            logger.info("Ensemble Top-3 AUC 11: %.4f vs Best Single Epoch AUC 11: %.4f", ens_score, best_single_score)
            recommended_path = top3_path if ens_score >= best_single_score else best_single_path

    if is_rank0:
        chosen_auc_11 = ens_score if (averaged_success and ens_score is not None) else (top_checkpoints[0][0] if top_checkpoints else None)
        valid_losses = [h["val_loss"] for h in history if math.isfinite(h.get("val_loss", float("nan")))]
        results_summary = {
            "history": history,
            "averaged_macro_auc_11": float(chosen_auc_11) if chosen_auc_11 is not None else None,
            "best_val_loss": float(min(valid_losses)) if valid_losses else None,
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
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=336)
    parser.add_argument("--crop-mm", type=float, default=DEFAULT_CROP_MM)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gold-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-cache", action="store_true", default=False)
    parser.add_argument("--cache-dir", type=str, default="/kaggle/tmp/rsna_cache_336_140mm")
    parser.add_argument("--weights-path", type=str, default=None)
    parser.add_argument("--allow-random-init", action="store_true", default=False)
    parser.add_argument("--slot-dropout-p", type=float, default=0.0)
    parser.add_argument("--soft-synovitis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weights-dir", type=str, default="outputs/weights")
    parser.add_argument("--tb-dir", type=str, default="outputs/tensorboard")
    parser.add_argument("--dicom-root", type=str, default=None)
    parser.add_argument("--series-csv", type=str, default=None)
    parser.add_argument("--splits-path", type=str, required=True)
    parser.add_argument("--labels-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, default="config.yaml")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    if not args.weights_path and not args.allow_random_init:
        raise ValueError("Must supply --weights-path or explicitly pass --allow-random-init.")

    # Pre-spawn validation on labels and splits files to fail fast in <1s before CUDA context or multiprocessing spawn
    labels_p = Path(args.labels_path)
    if not labels_p.is_file():
        raise FileNotFoundError(f"Missing labels file: {labels_p}")
    splits_p = Path(args.splits_path)
    if not splits_p.is_file():
        raise FileNotFoundError(f"Missing splits file: {splits_p}")

    chk_df = pd.read_parquet(labels_p) if labels_p.suffix == ".parquet" else pd.read_csv(labels_p, nrows=20)
    chk_splits = pd.read_parquet(splits_p) if splits_p.suffix == ".parquet" else pd.read_csv(splits_p, nrows=20)
    if "is_gold" not in chk_df.columns and "is_gold" not in chk_splits.columns:
        raise KeyError(f"Neither labels file ({labels_p}) nor splits file ({splits_p}) contains required 'is_gold' column.")
    if args.soft_synovitis and "synovitis_soft" not in chk_df.columns:
        raise KeyError(f"Labels file at {labels_p} missing required 'synovitis_soft' column for soft supervision.")

    if args.require_cache and args.cache_dir:
        splits_df = pd.read_csv(args.splits_path)
        uids = splits_df["StudyInstanceUID"].astype(str).tolist()
        valid_uids, empty_uids = verify_cache(args.cache_dir, uids, target_size=args.img_size, crop_mm=args.crop_mm)
        args.valid_uids = frozenset(valid_uids)
        args.empty_uids = frozenset(empty_uids)
        logger.info("Preflight cache verification passed for %d studies", len(valid_uids))

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if world_size > 1:
        torch.multiprocessing.spawn(train_ddp_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train_ddp_worker(local_rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()
