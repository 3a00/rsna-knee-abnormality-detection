# 02: Fast 6-Slot DICOM Preprocessing Pipeline

**What to build:** A production-grade DICOM dataset and preprocessing module that reconstructs true 3D spatial slice ordering via physical normal projection sorting ($k = \vec{p} \cdot \vec{n}$), extracts fixed 140 mm physical FOV crops normalized by `PixelSpacing`, routes series into 6 standardized anatomical contrast slots (Sag Fluid FS, Cor Fluid FS, Ax Fluid FS, Sag Fluid Non-FS, Cor T1, Sag T1), and forms 2.5D adjacent slice triplets ($\pm 1$). Slice ordering overhead across a study is reduced from minutes to sub-second speeds via a 16-worker `stop_before_pixels` multithreaded header scan.

**Blocked by:** 01: End-to-End Wiring Smoke Tracer

**Status:** completed

- [x] DICOM header reading parses `ImageOrientationPatient`, `ImagePositionPatient`, and `PixelSpacing` using `stop_before_pixels=True` in a 16-worker thread pool.
- [x] Slice ordering strictly follows the physical plane normal projection $k = \vec{p} \cdot (\vec{r} \times \vec{c})$ and passes monotonicity tests across multi-vendor test series.
- [x] Millimeter cropping extracts a square centered 140 mm window scaled by `PixelSpacing`, resized to 336×336 px, producing invariant anatomical scale across variable scanner resolutions.
- [x] Extracted study output tensor matches shape `(6, 3, 336, 336)` with an accompanying boolean presence mask of shape `(6,)` indicating slot availability.
- [x] Missing slots are populated with zeros without raising runtime exceptions or dropping studies.
- [x] Unit tests in `tests/test_efficiency_geometry.py` pass verifying normal projection ordering, FOV scale invariance, and 6-slot tensor shapes.
