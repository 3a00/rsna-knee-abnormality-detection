# RSNA Knee Abnormality Detection — Project Agent Rules

## Project Identity

- **Competition:** RSNA 2026 Knee Abnormality Detection (Kaggle Code Competition)
- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
- **Host:** Radiological Society of North America (RSNA)
- **Project Folder:** `/home/wenalz/Documents/Antigravity projects/RSNA_MRI/`
- **Deadline:** Final submission October 22, 2026 | Team merger deadline October 15, 2026
- **Prize Pool:** $77,000 (main leaderboard + efficiency track)
- **Evaluation Metric:** Macro-averaged ROC-AUC across 12 binary knee abnormality labels

---

## Important Warnings & Non-Negotiables

> [!CAUTION]
> **NEVER mix files from ThermoGuard-DFU (graduation project) into this folder.**
> Keep all RSNA work strictly in `/home/wenalz/Documents/Antigravity projects/RSNA_MRI/`.

> [!WARNING]
> **NEVER train models inside the submission notebook.** Only load pre-trained `.pth` weights and run offline inference.

> [!WARNING]
> **NEVER use a random split for cross-validation.** Random CV leaks scanner/site signatures (+0.053 artificial AUC). Always use Scanner-Fingerprint Stratified GroupKFold.

> [!IMPORTANT]
> **NEVER apply softmax to the final 12-label output.** Each label is an independent binary target — output raw logits and apply sigmoid per label.

> [!IMPORTANT]
> **NEVER hardcode absolute file paths** inside `src/` modules. Always reference `config.yaml` or function arguments.

---

## Repository Structure

```
RSNA_MRI/
├── AGENTS.md                   ← Project rules, guardrails, docs index
├── CONTEXT.md                  ← Ubiquitous domain model & glossary
├── config.yaml                 ← Global hyperparameters, paths, model settings
├── docs/                       ← Clinical, anatomical, and reference guidelines
│   ├── CLINICAL_GUIDELINES.md  ← Full MSK radiology guidelines, thresholds & phrasing
│   ├── adr/                    ← Architecture Decision Records
│   ├── agents/                 ← Issue tracker, triage, and domain consumer configs
│   └── discussions/            ← Kaggle community insights & technical benchmarks
├── notebooks/                  ← Kaggle-ready notebooks (eda, training, submission)
├── scripts/                    ← CLI verification and execution scripts
├── src/                        ← Modular Python source code (datasets, models, nlp, training, utils)
├── data/                       ← Raw inputs, processed labels, CV splits
├── outputs/                    ← Saved weights (.pth), training logs, submissions
└── .scratch/                   ← Feature workspaces and active issue tickets
```

---

## Development Roadmap & Status Summary

- **Phase 1 — EDA & Data Infrastructure:** ✅ Completed. DICOM loader with 3-priority classification, volume percentile normalization (0.5–99.5%), physics tag fallback.
- **Phase 2 — Weak Label Extraction from Reports:** ✅ Completed. Two-tier LLM extraction (Gemini 3.7 Flash → 3.6 Flash) with selective silence imputation (`synovitis_soft` 0.22/0.63, structural absence to 0, acute tears to NaN). 100% gold concordance, ~95% spot-checked accuracy. See [ADR-0001](docs/adr/0001-two-tier-llm-weak-label-extraction-and-imputation.md).
- **Phase 3 — Image Baseline:** 🏆 Baseline achieved (v16 3-Fold Mean Macro AUC: 0.8008 ± 0.0042 on Kaggle T4). 2.5D MIL architecture (EfficientNet-B2 + AttentionPooling). Active refinement in progress. See [ADR-0002](docs/adr/0002-2.5d-mil-with-attention-pooling.md), [ADR-0003](docs/adr/0003-single-gpu-execution-over-dataparallel.md), [ADR-0004](docs/adr/0004-tesla-t4-selection-over-p100.md).
- **Phase 4 — Multimodal Fusion:** ⏳ Upcoming. Multilingual text encoder (XLM-RoBERTa) + late/feature fusion with image model.
- **Phase 5 — Optimization & Ensembling:** ⏳ Upcoming. Multi-backbone ensembles (EfficientNet + ConvNeXt + DINOv2), calibration.
- **Phase 6 — Final Kaggle Submission:** ⏳ Upcoming. Self-contained offline notebook running in ≤9 hours.

---

## Coding Conventions & File Categorization

### Code Standards
- **Python 3.10+** & **PyTorch** throughout.
- **Type Hints** and **Google-style docstrings** on all public interfaces.
- **Single-Responsibility Modules**: one clear concern per file (`loader`, `model`, `loss`, `metrics`).
- **Reproducibility**: Set seeds (`torch`, `numpy`, `random`) at script initialization.
- **Model Output**: Always return raw logits of shape `(batch_size, 12)`. Sigmoid applied only at inference.

### File Categorization & Naming
1. **Production Core Files:** Reside strictly in `src/`, `scripts/`, `data/`, `notebooks/`, `docs/`, `outputs/` with clean permanent names.
2. **Temporary / Scratch / Probing Files:** Any exploratory script MUST be prefixed with `tmp_` or `scratch_`, start with `# [TEMPORARY / SCRATCH FILE — NOT FOR PRODUCTION]`, and can be safely deleted at any time.

---

## Gold Study Governance Rules

- **Statistical threshold:** At n=58 gold studies, differences below ~0.02 Macro AUC are statistical noise. Do not treat small deltas as meaningful signal or re-run experiments chasing sub-0.02 improvements.
- **No extraction overfitting:** Do NOT repeatedly tune Phase 2 extraction prompts against the 58 gold studies. Run concordance validation **once** after extraction is complete. The structural ceiling is ~82.5% (Discussion 05 audit) — reaching 83–85% is best-in-class performance. Do NOT target 90%+.
- **Gold utilization downstream:** Use gold studies for loss upweighting (5.0× in `MaskedBCEWithLogitsLoss`) and OOF Platt temperature calibration. Do NOT hold them out entirely from training.
- **Authoritative language distribution:** EN 28, ES 10, TR 6, HR 4, BG 3, EL 3, NL 2, DE 2. Use this for stratified concordance reporting — not the community-estimated EN 45/ES 5/OTHER 8 figures.
- **Image label is always authoritative:** When a clinical report and the image-derived gold label disagree, the image label wins. This is by competition design (two independent readers + adjudicator on 3D MRI review).

---

## DICOM Series Routing Rules (Physics-First)

When classifying DICOM series by sequence type and plane, **physics tags are primary; `SeriesDescription` text is secondary only**.

| Classification | Primary Signal | Secondary Signal |
|---|---|---|
| **Fluid-Sensitive vs. Anatomical** | `EchoTime (TE) > 30ms` → T2/fluid-sensitive; low TE + low TR → T1 | `SeriesDescription` keywords (T2, PD_FS, STIR, T1) as confirmation vote only |
| **Plane (Sagittal / Coronal / Axial)** | `ImageOrientationPatient` DICOM tag | `SeriesDescription` text plane keywords as secondary |

**Why:** Radiographers at 16 international institutions use wildly inconsistent `SeriesDescription` abbreviations across languages and vendors. Physics tags are standardized across all scanners.

**Action:** Spot-check a random sample of 20 routed series visually before training. A systematic routing error silently corrupts all downstream model training.

---

## Clinically Calibrated LLM Extraction Prompt

Used by the Phase 2 AGY multi-agent pipeline (`src/nlp/extractor_agent.py`). LLM API usage is permitted by competition rules (Section 2.6.b). Preferred: Google Gemini Flash.

```
You are an expert subspecialty musculoskeletal (MSK) radiologist. Read the following knee MRI
radiology report (written in any language) and extract the exact clinical status for each of
the 12 target findings below.

CRITICAL ANNOTATION RULES (Follow the official RSNA Radiologist Ground-Truth Guidelines):
1. Golden Rule: Borderline, mild, or ambiguous findings MUST be labeled as 0 (favor high specificity).
2. acl_tear: 1 ONLY if high-grade partial (>50% fibers disrupted) or full-thickness tear / rupture. Mild degeneration or intact reconstruction = 0.
3. mcl_tear: 1 ONLY if high-grade partial or complete tear. Low-grade sprain (Grade 1/2) or remote thickening = 0.
4. medial_meniscus: 1 ONLY if tear contacts articular surface or shows displacement/flap/bucket-handle. Intrasubstance degeneration = 0.
5. lateral_meniscus: Same criteria as medial meniscus. Intrasubstance degeneration = 0.
6. medial_oa: 1 ONLY if moderate-to-severe (>1cm area with >50% cartilage thickness loss). Mild thinning = 0.
7. lateral_oa: Same criteria as medial OA. Mild thinning = 0.
8. patellofemoral_oa: Same criteria as medial OA. Mild thinning = 0.
9. joint_effusion: 1 ONLY if moderate or large fluid distension. Trace or physiological fluid = 0.
10. synovitis: 1 ONLY if marked synovial thickening or inflammatory hypertrophy is described.
11. bakers_cyst: 1 ONLY if moderate or large popliteal cyst. Tiny incidental bursa fluid = 0.
12. bone_contusion: 1 ONLY for traumatic marrow edema / impact contusion. Chronic degenerative OA edema = 0.
13. fracture: 1 ONLY for acute cortical fracture line. Healed remote fracture = 0.

OUTPUT FORMAT: Output ONLY a valid JSON with these EXACT numeric values:
  1  = The report EXPLICITLY states this finding is present / positive.
  0  = The report EXPLICITLY states this finding is absent / normal / negative.
 -1  = The report does NOT address or mention this finding at all (silence / unaddressed).
(Do NOT use 0 for silence — use -1. Reserve 0 only for explicit negations.)
{
  "acl_tear": 0, "mcl_tear": 0, "medial_meniscus": 0, "lateral_meniscus": 0,
  "medial_oa": 0, "lateral_oa": 0, "patellofemoral_oa": 0, "joint_effusion": 0,
  "synovitis": 0, "bakers_cyst": 0, "bone_contusion": 0, "fracture": 0
}

Radiology Report:
{report_text}
```

**Operational rules:** Rate-limit API calls. Cache all responses to `data/labels/cache/{study_uid}.json`. Never re-call the API for the same study UID.

---

## Supervisor Notes

- The user (wenalz) acts as the **Architect and Supervisor** — they define strategy and review results.
- The AI agent writes all Python code, notebooks, scripts, and debugging fixes.
- Always explain design decisions clearly before implementing them.
- When proposing a new experiment or approach, present it as a short plan first and wait for approval.

---

## Agent Skills & Documentation Index

### Domain & Decisions
- **Domain Model & Glossary:** See [`CONTEXT.md`](CONTEXT.md) for clinical label definitions, 2-slice rule, and silence semantics.
- **Architecture Decision Records:** See [`docs/adr/`](docs/adr/) for rationale behind core irreversible design decisions.
- **MSK Clinical Guidelines:** See [`docs/CLINICAL_GUIDELINES.md`](docs/CLINICAL_GUIDELINES.md).

### Issue Tracking & Engineering Skills
- **Issue tracker:** Local Markdown in `.scratch/`. See [`docs/agents/issue-tracker.md`](docs/agents/issue-tracker.md).
- **Triage labels:** Canonical 5-role triage vocabulary. See [`docs/agents/triage-labels.md`](docs/agents/triage-labels.md).
- **Domain docs consumer rules:** Single-context layout. See [`docs/agents/domain.md`](docs/agents/domain.md).
