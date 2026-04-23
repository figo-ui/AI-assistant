# Healthcare AI Model Card

## Overview

This deployment package contains three production-ready ML models for a healthcare AI assistant:

| Task | Model | Val Macro F1 | Test Macro F1 | Baseline F1 | Improvement |
|------|-------|-------------|--------------|-------------|-------------|
| Triage Text Classification | SGD + TF-IDF | 0.8967 | 0.8831 | 0.7709 | **+0.1258** |
| Dialogue Intent Classification | SGD + TF-IDF | 0.9982 | 0.9989 | 0.9763 | **+0.0219** |
| Skin Lesion Image Classification | ImprovedDermCNN | 0.4741 | 0.4490 | 0.307 | **+0.1420** |

---

## Task 1: Triage Text Classification

### Model Details
- **Architecture**: SGDClassifier (modified_huber loss) + TF-IDF (word 1-2gram, 40k features)
- **Classes**: 73 medical conditions
- **Training data**: 24,366 samples (merged processed + unified datasets)
- **Preprocessing**: Text truncated to 50 words, strip_accents=unicode

### Performance
- Train Macro F1: 0.9102
- Val Macro F1: **0.8967** (+0.1258 vs baseline 0.7709)
- Test Macro F1: 0.8831
- Overfitting gap: 0.0135 (well-fitted)

### Worst-performing classes
- Acute rhinosinusitis (F1: 0.25) — overlaps with common cold symptoms
- Chronic rhinosinusitis (F1: 0.40) — similar to acute rhinosinusitis
- Bronchitis (F1: 0.51) — overlaps with URTI and pneumonia

### Artifacts
- `text_classifier.joblib` — SGDClassifier
- `tfidf_vectorizer.joblib` — TF-IDF vectorizer
- `text_labels.json` — 73 class names

---

## Task 2: Dialogue Intent Classification

### Model Details
- **Architecture**: SGDClassifier (modified_huber loss) + TF-IDF (word 1-3gram, 50k features)
- **Classes**: 15 medical dialogue intents
- **Training data**: 11,332 samples (MedQuAD dataset)
- **Intents**: causes, complications, considerations, exams and tests, frequency, genetic changes, information, inheritance, outlook, prevention, research, stages, susceptibility, symptoms, treatment

### Performance
- Train Macro F1: 1.0000
- Val Macro F1: **0.9982** (+0.0219 vs baseline 0.9763)
- Val Accuracy: 0.9994 (matches baseline)
- Overfitting gap: 0.0018 (well-fitted)

### Artifacts
- `dialogue_intent_classifier.joblib` — SGDClassifier
- `dialogue_intent_vectorizer.joblib` — TF-IDF vectorizer
- `dialogue_intent_labels.json` — 15 intent names

---

## Task 3: Skin Lesion Image Classification

### Model Details
- **Architecture**: ImprovedDermCNN (custom CNN with BatchNorm + Dropout)
- **Input**: 28×28 RGB images (DermaMNIST format)
- **Classes**: 7 skin lesion types
- **Training**: 30 epochs, cosine annealing LR, class-weighted loss (58x imbalance)
- **Augmentation**: Random flip (H/V), random 90° rotation

### Performance
- Train Macro F1: 0.4952
- Val Macro F1: **0.4741** (+0.1177 vs baseline 0.3564)
- Test Macro F1: 0.4490 (+0.1420 vs baseline 0.307)
- Test Accuracy: 0.5796 (+0.0774 vs baseline 0.5022)
- Overfitting gap: 0.0211 (well-fitted)

### Class Performance (Test Set)
| Class | F1 | Support |
|-------|-----|---------|
| actinic keratoses | 0.39 | 66 |
| basal cell carcinoma | 0.44 | 103 |
| benign keratosis-like lesions | 0.47 | 220 |
| dermatofibroma | 0.13 | 23 |
| melanoma | 0.40 | 223 |
| melanocytic nevi | 0.72 | 1341 |
| vascular lesions | 0.59 | 29 |

### Artifacts
- `skin_cnn_torch.pt` — PyTorch checkpoint
- `image_labels.json` — 7 class names

---

## Safety & Limitations

⚠️ **IMPORTANT MEDICAL DISCLAIMER**

These models are **NOT** intended for:
- Final clinical diagnosis
- Replacement of professional medical advice
- Unsupervised clinical decision-making
- Emergency triage without human oversight

These models ARE intended for:
- Preliminary symptom assessment support
- Routing patients to appropriate care
- Augmenting (not replacing) clinical judgment

### Known Limitations
1. **Triage**: Struggles with overlapping respiratory conditions (rhinosinusitis vs bronchitis)
2. **Image**: DermaMNIST is a small, low-resolution dataset (28×28). Real-world performance may differ significantly
3. **Dialogue**: Trained on MedQuAD (English only). Non-English queries may degrade performance

---

## Deployment

### Backend Compatibility
All artifacts are compatible with the existing Django/DRF backend:
- `text_classifier.joblib` → loaded by `backend/guidance/services/text_model.py`
- `dialogue_intent_classifier.joblib` → loaded by dialogue service
- `skin_cnn_torch.pt` → loaded by `backend/guidance/services/image_model.py`

### Inference Latency (CPU)
- Triage: ~5-20ms per request
- Dialogue: ~2-5ms per request
- Image: ~50-200ms per request (28×28 input)

### Monitoring Hooks
- Input validation: `_validate_text_input()` in `inference.py`
- Drift detection stub: `check_input_drift()` in `inference.py`
- Health check: `health_check()` in `inference.py`

---

## Training Pipeline

All training scripts are in `backend/scripts/`:
- `step1_eda.py` — Dataset EDA
- `step2_restructure.py` — Data restructuring
- `step3_train_triage_v4.py` — Triage model training
- `step3_train_dialogue_v3.py` — Dialogue model training
- `step3_train_image_v2.py` — Image model training
- `step4_analysis.py` — Performance analysis

Restructured datasets: `data/dataset_v1.0/`

---

*Generated by Timo ML Pipeline v2.0 | Healthcare AI Assistant*
