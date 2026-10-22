# [TEMPORARY / SCRATCH FILE — NOT FOR PRODUCTION]
"""scripts/scratch_time_inference.py

Gate 2 Timing Benchmark: Offline DICOM Decoding & ViT Forward Pass.
Evaluates end-to-end inference latency on 100 training studies under single-GPU execution.
Projects test-time inference duration across 1,300 studies on Dual Tesla T4 GPUs.
Pass threshold: Projected runtime <= 15.0 minutes (900 seconds).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.efficiency_pipeline import process_study_to_9slice_cache
from src.models.dinov2_slothead import DINOv2SlotHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def benchmark_inference(
    dicom_root: Path,
    study_uids: list[str],
    weights_path: Path | None = None,
    device_str: str = "cuda:0",
    target_studies: int = 100,
    n_models: int = 1,
) -> float:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info("Initializing DINOv2SlotHead on device: %s", device)
    model = DINOv2SlotHead(img_size=336, pretrained=False)
    if weights_path and weights_path.is_file():
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))
        logger.info("Loaded checkpoint from %s", weights_path)
    model.to(device)
    model.eval()

    sample_uids = study_uids[:min(target_studies, len(study_uids))]
    logger.info("Beginning online DICOM timing benchmark across %d studies...", len(sample_uids))

    t_decode = 0.0
    t_forward = 0.0
    n_done = 0

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
        # Warm-up (not timed)
        _ = model(torch.zeros(1, 6, 9, 336, 336, device=device), torch.ones(1, 6, device=device))
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for uid in sample_uids:
            study_dir = dicom_root / uid
            if not study_dir.is_dir():
                cands = list(dicom_root.glob(f"*/{uid}"))
                if not cands:
                    continue
                study_dir = cands[0]

            t_dec0 = time.time()
            tensor, mask = process_study_to_9slice_cache(study_dir, num_workers=2, target_size=336)
            t_decode += time.time() - t_dec0

            t_fwd0 = time.time()
            _ = model(tensor.unsqueeze(0).to(device).float(), mask.unsqueeze(0).to(device))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_forward += time.time() - t_fwd0
            n_done += 1

            if n_done % 25 == 0 or n_done == len(sample_uids):
                logger.info("Processed %d/%d studies (%.2fs/study)", n_done, len(sample_uids), (time.time() - t0) / n_done)

    total_time = time.time() - t0
    n_done = max(1, n_done)
    s_decode = t_decode / n_done
    s_forward = t_forward / n_done
    s_per_study = s_decode + n_models * s_forward

    projected_total_sec = s_per_study * 650.0
    projected_total_min = projected_total_sec / 60.0

    projected_decode_sec = s_decode * 650.0
    projected_decode_min = projected_decode_sec / 60.0

    governing_min = projected_total_min

    print("=" * 70)
    print("GATE 2 INFERENCE TIMING BENCHMARK REPORT")
    print("=" * 70)
    print(f"Ensemble Model Count:          {n_models}")
    print(f"Studies Evaluated:             {n_done}")
    print(f"Total Elapsed Time:            {total_time:.1f} s")
    print(f"  - DICOM Decode + Pre:        {t_decode:.1f} s ({s_decode*1000:.1f} ms/study) [num_workers=2]")
    print(f"  - ViT Forward Pass (1 model):{t_forward:.1f} s ({s_forward*1000:.1f} ms/study)")
    if n_models > 1:
        print(f"  - ViT Forward Pass ({n_models} models):{t_forward * n_models:.1f} s ({s_forward * n_models * 1000:.1f} ms/study)")
    print(f"Average Latency:               {s_per_study:.3f} s/study")
    print(f"Projected Total Time (2 GPUs): {projected_total_min:.2f} min ({projected_total_sec:.1f} s) [1300 studies]")
    print(f"Projected Decode Bound:        {projected_decode_min:.2f} min ({projected_decode_sec:.1f} s)")
    print(f"Governing Test Time:           {governing_min:.2f} min")
    print(f"Efficiency Track Budget:       15.00 min (900.0 s)")
    print("=" * 70)

    if governing_min <= 15.0:
        print("RESULT: PASS [Projected inference is within 15-minute budget]")
    else:
        print(f"RESULT: FAIL [Projected inference {governing_min:.2f} min exceeds 15.0 min limit]")
    return governing_min


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 2 DICOM timing benchmark")
    parser.add_argument("--dicom-root", type=str, required=True)
    parser.add_argument("--splits-path", type=str, required=True)
    parser.add_argument("--weights-path", type=str, default=None)
    parser.add_argument("--n-studies", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-models", type=int, default=1, help="Number of model checkpoints in ensemble (default: 1)")
    args = parser.parse_args()

    splits_df = pd.read_csv(args.splits_path)
    uids = splits_df["StudyInstanceUID"].astype(str).tolist()
    gov_min = benchmark_inference(
        dicom_root=Path(args.dicom_root),
        study_uids=uids,
        weights_path=Path(args.weights_path) if args.weights_path else None,
        device_str=args.device,
        target_studies=args.n_studies,
        n_models=args.n_models,
    )
    if gov_min > 15.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
