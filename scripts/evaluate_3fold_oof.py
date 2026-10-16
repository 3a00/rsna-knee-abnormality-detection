"""scripts/evaluate_3fold_oof.py

Full 3-Fold Out-of-Fold (OOF) Evaluator and Cross-Validation Benchmarking Engine.
Assembles out-of-fold predictions across studies from validation records,
resolves checkpoints from atomic summary markers, strictly validates model weights with strict=True,
evaluates Mean +/- Std (ddof=1) of per-fold Macro AUC, compares pooled raw logits against per-fold
percentile rank normalization, saves all benchmarks to disk first, and verifies the optional reproduction gate last.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

user_site = Path.home() / f".local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
if user_site.is_dir() and str(user_site) not in sys.path:
    sys.path.append(str(user_site))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.efficiency_pipeline import DEFAULT_CROP_MM, verify_cache
from src.models.dinov2_slothead import DINOv2SlotHead
from src.training.metrics import (
    compute_competition_metrics,
    load_label_columns,
    percentile_rank_transform,
)
from src.training.train_efficiency import EfficiencyStudyDataset, seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def locate_fold_checkpoint(weights_dir: Path, fold_idx: int) -> Path:
    """Find best checkpoint for given fold, inspecting fold{k}_summary.json first."""
    summary_p = weights_dir / f"fold{fold_idx}_summary.json"
    if summary_p.is_file():
        try:
            data = json.loads(summary_p.read_text(encoding="utf-8"))
            rec_name = data.get("recommended_weights")
            if rec_name and (weights_dir / rec_name).is_file():
                return weights_dir / rec_name
        except Exception as e:
            logger.debug("Could not read %s: %s", summary_p, e)

    candidates = [
        weights_dir / f"dinov2_slothead_fold{fold_idx}_best.pth",
        weights_dir / f"dinov2_slothead_fold{fold_idx}_top3.pth",
    ]
    if fold_idx == 0:
        candidates.extend([
            weights_dir / "dinov2_slothead_best_single.pth",
            weights_dir / "dinov2_slothead_top3.pth",
        ])

    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"Could not locate checkpoint for Fold {fold_idx} in {weights_dir}. Candidates tested: {[p.name for p in candidates]}"
    )


def compute_fold_predictions_inference(
    checkpoint_path: Path,
    val_df: pd.DataFrame,
    label_cols: list[str],
    cache_dir: str | Path,
    img_size: int = 336,
    crop_mm: float = DEFAULT_CROP_MM,
    batch_size: int = 16,
    device: torch.device = torch.device("cpu"),
) -> list[dict[str, Any]]:
    """Run offline evaluation inference on validation fold studies (fallback path)."""
    valid_uids, _ = verify_cache(cache_dir, val_df["StudyInstanceUID"].astype(str).tolist(), target_size=img_size, crop_mm=crop_mm)
    keep_set = set(valid_uids)
    n0 = len(val_df)
    val_df = val_df[val_df["StudyInstanceUID"].astype(str).isin(keep_set)].reset_index(drop=True)
    if len(val_df) < n0:
        logger.warning("Fallback inference: %d studies missing from cache, skipped", n0 - len(val_df))

    logger.info("Executing offline inference for %d studies using %s", len(val_df), checkpoint_path.name)
    val_dataset = EfficiencyStudyDataset(
        study_uids=val_df["StudyInstanceUID"].tolist(),
        study_dirs=[Path(cache_dir) / str(u) for u in val_df["StudyInstanceUID"]],
        labels=val_df[label_cols].values,
        is_gold=val_df["is_gold"].values if "is_gold" in val_df else None,
        cached_tensor_dir=cache_dir,
        target_size=img_size,
        crop_mm=crop_mm,
        is_train=False,
        require_cache=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = DINOv2SlotHead(pretrained=False, img_size=img_size).to(device)
    sd = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    model.load_state_dict(sd, strict=True)
    model.eval()

    records = []
    with torch.no_grad():
        for batch in val_loader:
            v_imgs = batch["image"].to(device, non_blocking=True).float()
            v_masks = batch["presence_mask"].to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                logits = model(v_imgs, v_masks)
            logits_np = logits.cpu().numpy()
            for i, uid in enumerate(batch["study_id"]):
                records.append({
                    "study_id": uid,
                    "logits": logits_np[i].tolist(),
                    "targets": batch["labels"][i].numpy().tolist(),
                    "is_gold": int(batch["is_gold"][i].item()),
                })
    return records


def load_or_compute_fold_records(
    fold_idx: int,
    weights_dir: Path,
    val_df: pd.DataFrame,
    label_cols: list[str],
    cache_dir: str | Path,
    img_size: int = 336,
    crop_mm: float = DEFAULT_CROP_MM,
    recompute: bool = False,
    device: torch.device = torch.device("cpu"),
) -> list[dict[str, Any]]:
    """Load cached validation prediction records, falling back to inference."""
    cached_records_file = weights_dir / f"val_records_fold{fold_idx}.json"
    if not recompute and cached_records_file.is_file():
        logger.info("Loading cached prediction records for Fold %d from %s", fold_idx, cached_records_file.name)
        with open(cached_records_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        return records

    ckpt_p = locate_fold_checkpoint(weights_dir, fold_idx)
    records = compute_fold_predictions_inference(
        checkpoint_path=ckpt_p,
        val_df=val_df,
        label_cols=label_cols,
        cache_dir=cache_dir,
        img_size=img_size,
        crop_mm=crop_mm,
        device=device,
    )
    with open(cached_records_file, "w", encoding="utf-8") as f:
        json.dump(records, f)
    return records


def assemble_oof_dataset(
    folds: list[int],
    splits_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    weights_dir: Path,
    cache_dir: str | Path,
    label_cols: list[str],
    img_size: int = 336,
    crop_mm: float = DEFAULT_CROP_MM,
    recompute: bool = False,
    device: torch.device = torch.device("cpu"),
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble complete out-of-fold predictions and apply per-fold percentile ranking.

    Returns:
        tuple of (oof_df, raw_logits_all, ranked_probs_all, targets_all)
    """
    fold_col = "fold_id" if "fold_id" in splits_df.columns else "fold"
    merged = splits_df.merge(labels_df, on="StudyInstanceUID", how="inner")
    if "is_gold" not in merged.columns and "is_gold_x" in merged.columns:
        merged["is_gold"] = merged["is_gold_x"]

    fold_dfs = []
    for f_idx in folds:
        val_df = merged[merged[fold_col] == f_idx].reset_index(drop=True)
        records = load_or_compute_fold_records(
            fold_idx=f_idx,
            weights_dir=weights_dir,
            val_df=val_df,
            label_cols=label_cols,
            cache_dir=cache_dir,
            img_size=img_size,
            crop_mm=crop_mm,
            recompute=recompute,
            device=device,
        )

        rec_dict = {str(r["study_id"]): r for r in records}
        val_df = val_df[val_df["StudyInstanceUID"].astype(str).isin(rec_dict)].reset_index(drop=True)

        val_logits = np.array([rec_dict[str(u)]["logits"] for u in val_df["StudyInstanceUID"]])

        try:
            val_ranked = percentile_rank_transform(val_logits)
        except ValueError as r_err:
            nan_mask = ~np.isfinite(val_logits)
            bad_indices = np.where(nan_mask.any(axis=1))[0]
            bad_uids = [val_df["StudyInstanceUID"].iloc[i] for i in bad_indices[:5]]
            raise ValueError(f"Fold {f_idx} ranking failed: non-finite logits in studies {bad_uids}: {r_err}") from r_err

        sub_df = pd.DataFrame({"StudyInstanceUID": val_df["StudyInstanceUID"], "fold_id": f_idx})
        sub_df["is_gold"] = val_df["is_gold"].astype(int)

        for i, col in enumerate(label_cols):
            sub_df[f"target_{col}"] = val_df[col].values
            sub_df[f"logit_{col}"] = val_logits[:, i]
            sub_df[f"rank_{col}"] = val_ranked[:, i]

        fold_dfs.append(sub_df)

    oof_df = pd.concat(fold_dfs, ignore_index=True)

    targets_all = oof_df[[f"target_{c}" for c in label_cols]].values
    raw_logits_all = oof_df[[f"logit_{c}" for c in label_cols]].values
    ranked_probs_all = oof_df[[f"rank_{c}" for c in label_cols]].values

    return oof_df, raw_logits_all, ranked_probs_all, targets_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Full 3-Fold OOF Evaluator & Cross-Validation Benchmark")
    parser.add_argument("--weights-dir", type=str, default="outputs/weights")
    parser.add_argument("--splits-path", type=str, default="data/splits/cv_folds_3fold.csv")
    parser.add_argument("--labels-path", type=str, default="data/labels/pseudo_labels.csv")
    parser.add_argument("--config-path", type=str, default="config.yaml")
    parser.add_argument("--cache-dir", type=str, default="/kaggle/tmp/rsna_cache_336_140mm")
    parser.add_argument("--img-size", type=int, default=336)
    parser.add_argument("--crop-mm", type=float, default=DEFAULT_CROP_MM)
    parser.add_argument("--output-csv", type=str, default="outputs/weights/oof_predictions.csv")
    parser.add_argument("--output-json", type=str, default="outputs/weights/oof_results.json")
    parser.add_argument("--expect-fold0-auc", type=float, default=None, help="Optional Fold 0 reproduction gate threshold")
    parser.add_argument("--recompute-preds", action="store_true", default=False)
    args = parser.parse_args()

    seed_everything(42)

    weights_dir = Path(args.weights_dir)
    splits_p = Path(args.splits_path)
    labels_p = Path(args.labels_path)

    splits_df = pd.read_parquet(splits_p) if splits_p.suffix == ".parquet" else pd.read_csv(splits_p)
    labels_df = pd.read_parquet(labels_p) if labels_p.suffix == ".parquet" else pd.read_csv(labels_p)
    label_cols = load_label_columns(args.config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = [0, 1, 2]

    logger.info("Assembling 3-Fold Out-of-Fold predictions across studies...")
    oof_df, raw_logits, ranked_probs, targets = assemble_oof_dataset(
        folds=folds,
        splits_df=splits_df,
        labels_df=labels_df,
        weights_dir=weights_dir,
        cache_dir=args.cache_dir,
        label_cols=label_cols,
        img_size=args.img_size,
        crop_mm=args.crop_mm,
        recompute=args.recompute_preds,
        device=device,
    )

    gold_mask = oof_df["is_gold"].values == 1

    # Optional Fold 0 gate evaluation (recorded without crashing output generation)
    f0 = oof_df[oof_df["fold_id"] == 0]
    gate_result = {"expected": args.expect_fold0_auc, "got": None, "passed": True}
    if len(f0) > 0 and args.expect_fold0_auc is not None:
        m0 = compute_competition_metrics(
            logits=f0[[f"logit_{c}" for c in label_cols]].values,
            targets=f0[[f"target_{c}" for c in label_cols]].values,
            gold_mask=(f0["is_gold"].values == 1),
            label_cols=label_cols,
        )
        gate_result["got"] = float(m0.macro_auc_11)
        logger.info("Fold 0 Validation Macro AUC 11: %.4f (Expected: %.4f)", m0.macro_auc_11, args.expect_fold0_auc)
        if abs(m0.macro_auc_11 - args.expect_fold0_auc) > 0.005:
            gate_result["passed"] = False
            logger.error(
                "Fold 0 reproduction gate FAILED: got %.4f vs expected %.4f (+/-0.005).",
                m0.macro_auc_11, args.expect_fold0_auc
            )

    # Compute Per-Fold CV Statistics with ddof=1 (sample standard deviation)
    per_fold_stats = {}
    for f_idx, group in oof_df.groupby("fold_id"):
        g_metrics = compute_competition_metrics(
            logits=group[[f"logit_{c}" for c in label_cols]].values,
            targets=group[[f"target_{c}" for c in label_cols]].values,
            gold_mask=(group["is_gold"].values == 1),
            label_cols=label_cols,
        )
        per_fold_stats[int(f_idx)] = {
            "macro_auc_11": g_metrics.macro_auc_11,
            "macro_auc_12": g_metrics.macro_auc_12,
            "n_valid_labels": g_metrics.n_valid_labels,
        }

    fold_aucs_11 = [v["macro_auc_11"] for v in per_fold_stats.values()]
    fold_aucs_12 = [v["macro_auc_12"] for v in per_fold_stats.values()]
    cv_mean_11 = float(np.mean(fold_aucs_11))
    cv_std_11 = float(np.std(fold_aucs_11, ddof=1)) if len(fold_aucs_11) > 1 else 0.0
    cv_mean_12 = float(np.mean(fold_aucs_12))
    cv_std_12 = float(np.std(fold_aucs_12, ddof=1)) if len(fold_aucs_12) > 1 else 0.0

    # Evaluate Pooled Raw Sigmoid OOF
    metrics_raw = compute_competition_metrics(
        logits=raw_logits,
        targets=targets,
        gold_mask=gold_mask,
        label_cols=label_cols,
    )
    if metrics_raw.n_valid_labels != 12:
        logger.warning("Pooled OOF has %d valid labels (expected 12)", metrics_raw.n_valid_labels)

    # Evaluate Pooled Per-Fold Percentile Rank Normalized OOF
    metrics_rank = compute_competition_metrics(
        logits=ranked_probs,
        targets=targets,
        gold_mask=gold_mask,
        label_cols=label_cols,
    )

    # Gold 58 Subset Evaluation
    gold_raw_11, gold_raw_12 = float("nan"), float("nan")
    gold_rank_11, gold_rank_12 = float("nan"), float("nan")
    if gold_mask.any():
        gold_m_raw = compute_competition_metrics(
            logits=raw_logits[gold_mask],
            targets=targets[gold_mask],
            gold_mask=np.ones(gold_mask.sum(), dtype=np.int64),
            label_cols=label_cols,
        )
        gold_m_rank = compute_competition_metrics(
            logits=ranked_probs[gold_mask],
            targets=targets[gold_mask],
            gold_mask=np.ones(gold_mask.sum(), dtype=np.int64),
            label_cols=label_cols,
        )
        gold_raw_11, gold_raw_12 = gold_m_raw.macro_auc_11, gold_m_raw.macro_auc_12
        gold_rank_11, gold_rank_12 = gold_m_rank.macro_auc_11, gold_m_rank.macro_auc_12

    logger.info("==================================================")
    logger.info("3-FOLD CROSS-VALIDATION BENCHMARK (N=%d Studies)", len(oof_df))
    logger.info("==================================================")
    for f_idx, stats in per_fold_stats.items():
        logger.info("  Fold %d: Macro AUC 11 = %.4f | Macro AUC 12 = %.4f (labels: %d)",
                    f_idx, stats["macro_auc_11"], stats["macro_auc_12"], stats["n_valid_labels"])
    logger.info("--------------------------------------------------")
    logger.info("CV Headline (Mean +/- Std, ddof=1): Macro AUC 11: %.4f +/- %.4f | Macro AUC 12: %.4f +/- %.4f",
                cv_mean_11, cv_std_11, cv_mean_12, cv_std_12)
    logger.info("--------------------------------------------------")
    logger.info("Pooled Raw Sigmoid OOF:     Macro AUC 11: %.4f | Macro AUC 12: %.4f | Gold: %.4f / %.4f",
                metrics_raw.macro_auc_11, metrics_raw.macro_auc_12, gold_raw_11, gold_raw_12)
    logger.info("Pooled Rank-Normalized OOF: Macro AUC 11: %.4f | Macro AUC 12: %.4f | Gold: %.4f / %.4f",
                metrics_rank.macro_auc_11, metrics_rank.macro_auc_12, gold_rank_11, gold_rank_12)
    logger.info("Calibration Rank Delta:     AUC 11: %+0.4f | AUC 12: %+0.4f",
                metrics_rank.macro_auc_11 - metrics_raw.macro_auc_11,
                metrics_rank.macro_auc_12 - metrics_raw.macro_auc_12)
    logger.info("==================================================")

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(args.output_csv, index=False)
    logger.info("Saved complete OOF prediction matrix to %s", args.output_csv)

    results_data = {
        "n_studies": len(oof_df),
        "cv_stats": {
            "mean_macro_auc_11": cv_mean_11,
            "std_macro_auc_11": cv_std_11,
            "mean_macro_auc_12": cv_mean_12,
            "std_macro_auc_12": cv_std_12,
            "std_ddof": 1,
            "per_fold": per_fold_stats,
        },
        "pooled_raw_oof": {
            "macro_auc_11": metrics_raw.macro_auc_11,
            "macro_auc_12": metrics_raw.macro_auc_12,
            "gold_auc_11": gold_raw_11,
            "gold_auc_12": gold_raw_12,
            "n_valid_labels": metrics_raw.n_valid_labels,
        },
        "pooled_rank_oof": {
            "macro_auc_11": metrics_rank.macro_auc_11,
            "macro_auc_12": metrics_rank.macro_auc_12,
            "gold_auc_11": gold_rank_11,
            "gold_auc_12": gold_rank_12,
        },
        "calibration_delta": {
            "auc_11": float(metrics_rank.macro_auc_11 - metrics_raw.macro_auc_11),
            "auc_12": float(metrics_rank.macro_auc_12 - metrics_raw.macro_auc_12),
        },
        "fold0_gate": gate_result,
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    logger.info("Saved OOF evaluation benchmark to %s", args.output_json)

    if not gate_result["passed"]:
        logger.error("Aborting with exit code 2 due to failed Fold 0 reproduction gate.")
        sys.exit(2)


if __name__ == "__main__":
    main()
