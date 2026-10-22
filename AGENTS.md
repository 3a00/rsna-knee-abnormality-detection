# RSNA Knee Abnormality Detection — Project Agent Rules

## Project Identity

- **Competition:** RSNA 2026 Knee Abnormality Detection (Kaggle Code Competition)
- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
- **Project Folder:** `/home/wenalz/Documents/Antigravity projects/RSNA_MRI/`
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

> [!CAUTION]
> **NEVER push `private-work` to GitHub (`origin`).**
> - The GitHub remote (`origin/main`) is public.
> - `main` tracks ONLY clean code (`src/`, `scripts/`, `tests/`, `config.yaml`, `CONTEXT.md`, `.gitignore`). All data, outputs, notebooks, docs, and scratch files are ignored and scrubbed from history.
> - `private-work` is strictly for local development and tracking of full files.
> - Sync to GitHub ONLY via:
>   `git checkout main && git checkout private-work -- src scripts tests config.yaml && git commit -m "sync" && git push origin main && git checkout private-work`

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

## Supervisor Notes

- The user (wenalz) acts as the **Architect and Supervisor** — they define strategy and review results.
- The AI agent writes all Python code, notebooks, scripts, and debugging fixes.
- When proposing a new experiment or approach, present it as a short plan first and wait for approval.
- **Review Packaging (`/package`):** When asked to package an issue for review and write or implement code for it, write the complete proposed implementation *strictly inside the review document* (`docs/llm review/`) using the requested skill (e.g. `yagni full`). Do **not** implement, touch, or run code in the active repository until the review feedback has been received and approved.

---

## Documentation Index

### Domain & Decisions
- **Domain Model & Glossary:** See [`CONTEXT.md`](CONTEXT.md) for clinical label definitions, 2-slice rule, and silence semantics.
- **Gold Study Governance:** See [`docs/GOLD_STUDIES.md`](docs/GOLD_STUDIES.md) for n=58 validation rules, language distribution, and loss weighting.
- **Architecture Decision Records:** See [`docs/adr/`](docs/adr/) for rationale behind core irreversible design decisions.
- **MSK Clinical Guidelines:** See [`docs/CLINICAL_GUIDELINES.md`](docs/CLINICAL_GUIDELINES.md).

### Issue Tracking & Engineering
- **Issue tracker:** Local Markdown in `.scratch/`. See [`docs/agents/issue-tracker.md`](docs/agents/issue-tracker.md).
- **Triage labels:** Canonical 5-role triage vocabulary. See [`docs/agents/triage-labels.md`](docs/agents/triage-labels.md).
- **Domain docs consumer rules:** Single-context layout. See [`docs/agents/domain.md`](docs/agents/domain.md).
