# Text Model Card

## Overview

Runtime text triage uses a classical symptom model:

- `TF-IDF`
- optional `TruncatedSVD`
- `LogisticRegression` or `XGBoost`

Primary artifacts:

- `text_classifier.joblib`
- `tfidf_vectorizer.joblib`
- `text_labels.json`
- `tfidf_svd.joblib` (optional)

## Intended Use

- symptom-to-condition ranking
- risk-routing support when combined with safety overrides
- fallback path when the optional LLM adapter is unavailable

## Not Intended Use

- final diagnosis
- unsupervised clinical decision-making
- specialist pathways without dedicated validation

## Training Data Requirements

- symptom-only narratives
- no target leakage (`Reason:`, medication lists, allergies, administrative encounter text)
- avoid generic labels such as `Condition 275`

## Evaluation Status

Current checked-in metrics are in `text_training_metrics.json`.

- accuracy: `0.9752`
- macro-F1: `0.7553`
- classes: `113`

Interpretation: this is a strong baseline for a prototype, but class imbalance is still material and macro-F1 is the more honest number.

## Runtime Controls

- safety override layer can promote emergency differentials
- bilingual normalization maps supported Amharic phrases into model-friendly English keywords
- real-time search does not change the local classifier; it augments context only

## Related Tooling

- preprocessing: `backend/scripts/preprocess_and_train.py`
- external-style evaluation: `backend/scripts/evaluate_text_model.py`
- regression suite: `backend/scripts/run_triage_regression.py`
- strict-JSON LLM dataset prep: `backend/scripts/prepare_triage_llm_dataset.py`
