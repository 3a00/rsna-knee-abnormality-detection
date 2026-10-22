# filepath: src/utils/seed.py
"""src/utils/seed.py

Reproducibility seed utilities for RSNA Knee MRI Efficiency Track.
Standalone module to prevent circular imports between datasets and training engines.
"""

from __future__ import annotations

import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set random seeds across Python, NumPy, PyTorch CPU and CUDA for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
