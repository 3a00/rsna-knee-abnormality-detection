# Two-Tier LLM Weak Label Extraction and Silence Imputation

Only 58 of 4,407 training studies contain verified gold labels; the remaining 4,349 must be derived from multi-lingual free-text reports. We adopted a two-tier LLM extraction pipeline (Gemini 3.7 Flash with Gemini 3.6 Flash fallback) using Pydantic schemas and clinically calibrated silence imputation (selective continuous soft imputation for Synovitis and structural absence mapping to 0 for OA/Baker's cyst, while masking unmentioned acute tears as NaN).

## Status

accepted

## Context

Radiology reports in the dataset span 9+ languages (English, Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch, French). Classical rule-based NLP or single-language regex cannot handle international terminology, varying linguistic negations, or nuanced clinical thresholds (e.g. distinguishing trace fluid from effusion). Furthermore, unmentioned findings carry distinct clinical meanings: absence of comment on chronic degenerative conditions implies normality (0), whereas unmentioned acute focal tears (ACL, meniscus) may simply be unaddressed.

## Considered Options

- **Naive Multi-Lingual Regex / Keyword Matching**: Low latency, but failed on non-English grammatical negation and subtle clinical gradations.
- **Global Zero-Imputation (`fillna(0)`)**: Assumed silence always equals negative, injecting severe false-negative bias into acute focal pathology classes.
- **Two-Tier LLM Extraction with Selective Imputation (Selected)**: Enforces 3-state output (`1`, `0`, `-1`), verifies 100% concordance on the 58 gold studies, applies continuous soft target (0.22 / 0.63) to unmentioned Synovitis based on Effusion correlation, maps structural absence findings to 0, and preserves unaddressed acute pathologies as loss-masked NaNs.

## Consequences

- 4,349 studies pseudo-labeled with ~95% estimated label-level accuracy across 8+ languages.
- Loss function must support NaN masking (`MaskedBCEWithLogitsLoss`) and continuous soft targets for Synovitis.
- Gold studies remain strictly protected from pseudo-label overwriting.
