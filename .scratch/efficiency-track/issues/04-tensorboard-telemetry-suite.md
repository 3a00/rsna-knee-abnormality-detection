# 04: TensorBoard Telemetry Suite & Diagnostic Visualizers

**What to build:** A comprehensive experiment tracking module (`src/utils/tensorboard.py`) providing a unified `TensorBoardLogger` interface (with a fallback no-op protocol) writing events to `outputs/tensorboard/<run_tag>/`. It records the computational model graph (`writer.add_graph`), training step loss, epoch validation loss, backbone and head learning rate schedules, overall Macro ROC-AUC, 12 individual finding ROC-AUC curves (with Synovitis evaluated strictly on Gold Studies), finding-to-slot attention heatmaps, and validation precision-recall (PR) curves.

**Blocked by:** 01: End-to-End Wiring Smoke Tracer

**Status:** completed
- [x] `TensorBoardLogger` and `NoOpLogger` implement a shared protocol so consumers can log without coupling to TensorBoard availability.
- [x] Model architecture graph is logged via `writer.add_graph` using a dummy `(1, 6, 3, 336, 336)` tensor without crashing or serialization errors.
- [x] Scalars for training step loss, epoch train/val loss, learning rates, and Macro ROC-AUC are logged and readable in the TensorBoard dashboard.
- [x] Per-label ROC-AUC scalars are logged individually for all 12 abnormalities, enforcing Gold Study filtering for Synovitis and non-NaN masking for other labels.
- [x] At validation epoch end, attention weights across the 12 findings and 6 slots are rendered into a labeled matplotlib heatmap figure and logged via `writer.add_figure`.
- [x] Precision-recall curves for each finding are logged at epoch completion via `writer.add_pr_curve`.
- [x] Unit tests in `tests/test_tensorboard_logging.py` verify all logging methods write without error and handle edge cases (e.g., all-zero predictions, missing labels).
