# Tesla T4 Target Hardware Selection Over Deprecated P100

All model training and verification notebooks must target Nvidia Tesla T4 (`sm_75`) GPUs, completely avoiding Tesla P100 (`sm_60`) hardware due to PyTorch 2.6+ dropping Pascal architecture CUDA kernel support.

## Status

accepted

## Context

Kaggle's environment runs Python 3.12 with PyTorch 2.6+. PyTorch 2.6+ dropped compiled CUDA kernels for Pascal architecture (`sm_60`), triggering immediate `CUDA error: no kernel image is available for execution on the device` crashes when executed on Tesla P100 GPUs (observed in Phase 3 v14). Furthermore, Kaggle has announced the complete retirement of P100 instances on September 15, 2026.

## Considered Options

- **PyTorch Downgrade in Notebook (`torch==2.2.2+cu118`)**: Restores Pascal support on P100, but introduces notebook setup latency and will become obsolete upon P100 retirement.
- **Tesla T4 Selection with Fail-Fast GPU Guard (Selected)**: Select T4 in Kaggle accelerator settings and insert an assertion check in notebook Cell 1 that checks `compute_capability >= (7, 0)` before launching training.

## Consequences

- Zero CUDA version compatibility errors across training and inference runs.
- Automatic safety net prevents burning Kaggle GPU quota on improperly configured sessions.
