"""
run_phase2_extraction.py -- Phase 2 entry-point script.

Launches the AGY Orchestrator agent which spawns N Extractor subagents
in parallel to extract 3-state weak labels from 4,349 multilingual reports.

Usage:
  export GEMINI_API_KEY=your_key_here
  python scripts/run_phase2_extraction.py
  python scripts/run_phase2_extraction.py --dry-run 100
  python scripts/run_phase2_extraction.py --skip-validation
  python scripts/run_phase2_extraction.py --workers 16

Environment Variables:
  GEMINI_API_KEY  -- Required (or pass --api-key)
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp import run_orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RSNA Phase 2: Weak label extraction via AGY multi-agent pipeline"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Gemini API key (default: reads GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--dry-run", type=int, default=None, metavar="N",
        help="Test mode: extract only the first N studies"
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip Validator subagent (use for incremental re-runs)"
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of concurrent Extractor subagents (default: 8)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run_orchestrator(
            config_path=args.config,
            api_key=args.api_key,
            dry_run_n=args.dry_run,
            skip_validation=args.skip_validation,
            n_concurrent_workers=args.workers,
        )
    )
