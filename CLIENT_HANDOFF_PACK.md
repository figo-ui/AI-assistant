# AI Healthcare Assistant Chatbot - Final Client Handoff Pack

Generated on: March 7, 2026

## 1. Delivery Summary

This delivery includes:

- Full-stack healthcare assistant chatbot (Django backend + React frontend)
- JWT authentication and profile management
- Chat sessions, saved chat history, and export (JSON/CSV)
- Symptom analysis pipeline:
  - Text model (retrained on rebalanced merged datasets with clinical override layer)
  - Dialogue intent model (retrained on MedQuAD + expanded open medical QA dialogue set)
  - Image model (trained on free DermaMNIST dataset)
- Risk scoring and prevention guidance
- Facility search and emergency contacts
- Admin APIs (users, facilities, analytics, config action)

## 2. Deployment Steps

### 2.1 Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- Windows/Linux/macOS supported

### 2.2 Backend Deployment

From project root:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Seed or update admin user:

```bash
python manage.py seed_admin --username admin --email admin@example.com --password StrongPass123 --reset-password
```

Backend base URL:

- `http://127.0.0.1:8000/api/v1`

### 2.3 Frontend Deployment

From project root:

```bash
cd frontend
npm install
npm run dev
```

Frontend local URL:

- `http://127.0.0.1:5173`

Production build:

```bash
npm run build
npm run preview
```

### 2.4 Required/Recommended Environment Variables (Backend)

- `DJANGO_SECRET_KEY` (required for production; use 32+ chars)
- `DJANGO_DEBUG` (`false` in production)
- `DJANGO_ALLOWED_HOSTS` (set production hostnames)
- `CORS_ALLOWED_ORIGINS` (set frontend production origin)
- `DB_ENGINE`, `DB_NAME` (switch from SQLite to MySQL/PostgreSQL for production)
- `GOOGLE_MAPS_API_KEY` (recommended for live Places results)
- `GOOGLE_PLACES_RADIUS_METERS` (default radius for Places)
- `TEXT_MODEL_PATH`, `TFIDF_VECTORIZER_PATH`, `TEXT_LABELS_PATH`
- `TEXT_SVD_PATH` (optional; only needed for SVD-backed text models)
- `DIALOGUE_INTENT_MODEL_PATH`, `DIALOGUE_INTENT_VECTORIZER_PATH`, `DIALOGUE_RESPONSE_TEMPLATES_PATH`
- `IMAGE_MODEL_PATH` (TensorFlow optional)
- `IMAGE_TORCH_MODEL_PATH` (active model path)
- `IMAGE_LABELS_PATH`, `IMAGE_INPUT_SIZE`

## 3. API List (Final)

Base path: `/api/v1/`

### 3.1 Health

- `GET /health/`

### 3.2 Authentication

- `POST /auth/register/`
- `POST /auth/login/`
- `POST /auth/refresh/`
- `POST /auth/logout/`

### 3.3 Profile

- `GET /profile/`
- `PATCH /profile/`

### 3.4 Chat

- `GET /chat/sessions/`
- `POST /chat/sessions/`
- `GET /chat/sessions/{session_id}/messages/`
- `POST /chat/sessions/{session_id}/analyze/` (multipart, supports image)
- `GET /chat/history/`
- `GET /chat/export/?format=json|csv`

### 3.5 Public Analysis

- `POST /analyze/` (multipart, consent required)

### 3.6 Location

- `GET /location/nearby/`
- `GET /location/directions/`
- `GET /location/emergency/`

### 3.7 Admin

- `GET /admin/users/`
- `PATCH /admin/users/{user_id}/`
- `GET /admin/facilities/`
- `POST /admin/facilities/`
- `PATCH /admin/facilities/{facility_id}/`
- `DELETE /admin/facilities/{facility_id}/`
- `GET /admin/analytics/`
- `POST /admin/config/`

### 3.8 Export

- `GET /export/profile/`

## 4. Model Metrics and Datasets

### 4.1 Text Model (Active)

Model artifacts:

- `backend/models/text_classifier.joblib`
- `backend/models/tfidf_vectorizer.joblib`
- `backend/models/text_labels.json`
- `backend/models/condition_name_map.json`
- `backend/models/class_distribution_train_raw.json`
- `backend/models/class_distribution_train_rebalanced.json`
- `backend/models/triage_regression_report.json`

Training metrics (`backend/models/text_training_metrics.json`):

- Accuracy: `0.9752`
- Macro F1: `0.7568`
- Samples: `28,169`
- Classes: `113`
- Model type: `LogisticRegression`
- Rebalance mode: `under_over`
- Majority cap: `COVID-19` / `Suspected COVID-19` capped to `15%` setting
- Minority target: `100` samples per class target during oversampling
- Calibration: `sigmoid` calibration enabled

Integrated dataset sources:

- `data/disease_symptom_processed.csv`
- `data/synthea_symptom_condition.csv`
- `data/kaggle/processed/kaggle_disease_symptom_processed.csv` (CC-BY-SA-4.0)
- `data/expanded_symptom_condition.csv` (expanded merge with DDXPlus + MedQA-derived diagnosis rows)

Integrated training outputs:

- `data/integrated_important_symptom_condition.csv`
- `data/kaggle/processed/integrated_important_plus_kaggle.csv`
- `data/expanded_symptom_condition_processed_min5_rebalanced.csv`

Clinical control layer added on top of model probabilities:

- Stroke / brain hemorrhage emergency override
- Acute coronary syndrome override
- Urinary tract / kidney infection override
- Contact-allergy / drug-reaction override for chemical rash patterns
- Pneumonia override for cough + fever + pleuritic chest pain
- Viral URI override for sore-throat/runny-nose conversational cases
- Strong non-respiratory COVID suppression

### 4.2 Dialogue Model (Active)

Model artifacts:

- `backend/models/dialogue_intent_classifier.joblib`
- `backend/models/dialogue_intent_vectorizer.joblib`
- `backend/models/dialogue_intent_labels.json`
- `backend/models/dialogue_response_templates.json`
- `backend/models/dialogue_training_metrics.json`

Dataset:

- MedQuAD (free medical QA dialogue corpus, CC BY 4.0)
- Expanded open dialogue/QA additions from MedMCQA and PubMedQA processing
- Built merged dataset CSV: `data/dialogue/expanded_medical_dialogue.csv`
- Dataset summaries:
  - `data/dialogue/medquad_dialogue_summary.json`
  - `data/expanded_symptom_condition_summary.json`

Training metrics (`backend/models/dialogue_training_metrics.json`):

- Accuracy: `0.9825`
- Macro F1: `0.9763`
- Weighted F1: `0.9832`
- Samples: `37,089`
- Intent classes: `17`

### 4.3 Image Model (Active)

Model artifacts:

- `backend/models/skin_cnn_torch.pt`
- `backend/models/image_labels.json`
- `backend/models/image_training_metrics.json`

Dataset:

- DermaMNIST (free public benchmark) in `data/image_datasets`

Training metrics (`backend/models/image_training_metrics.json`):

- Dataset: `dermamnist`
- Classes: `7`
- Train/Val/Test: `7007 / 1003 / 2005`
- Best validation macro F1: `0.3564`
- Test macro F1: `0.3070`
- Test accuracy: `0.5022`
- Input size: `64`

## 5. Acceptance/Readiness Checks Completed

- Backend checks: `python manage.py check` passed
- Deployment security check: `python manage.py check --deploy --fail-level ERROR` completed with warnings only
- Frontend build: `npm run build` passed
- Text + dialogue retraining pipeline completed
- Database migrations applied
- API smoke checks passed:
  - Auth register/login
  - Chat session create
  - Analyze endpoint
  - Image inference returns non-empty predictions (`image-cnn-torch-v1`)
- Text triage regression smoke test passed:
  - `backend/models/triage_regression_report.json`
  - `8/8` targeted cases matched expected top-3 and urgency pattern

## 6. Known Limitations

1. Medical disclaimer:
- System output is decision support only, not a clinical diagnosis.

2. Image model quality:
- Current DermaMNIST model performance is moderate and not clinical-grade.

3. Dataset license note:
- DermaMNIST/HAM10000 includes non-commercial license terms (CC BY-NC 4.0 context).
- Commercial deployment requires legal/license review or a commercially permitted alternative dataset.
- Kaggle `medical-chatbot-dataset` license is currently `unknown`; it must not be used in production/client delivery until explicitly clarified.
- Experimental unknown-license Kaggle dialogue data is excluded from the current production-safe handoff recommendation.

4. Class imbalance and differential ranking:
- Source data still began with heavy COVID skew; rebalancing and clinical overrides materially reduce the bias, but this is not equivalent to having a large real-world triage corpus.
- Some classes remain synthetic or weakly represented even after expansion; uncommon differentials can still rank noisily outside the top result.

5. Facility search behavior:
- Without `GOOGLE_MAPS_API_KEY`, system falls back to local facility registry data only.

6. Dialogue dataset scope:
- MedQuAD / MedMCQA / PubMedQA improve medical language coverage, but they are still QA-style corpora rather than real nurse-triage conversations.
- A future LLM fine-tuning stage is still recommended for stronger reasoning and more natural multi-turn conversation quality.

7. Production hardening pending:
- No full production deployment stack included (reverse proxy, TLS, CI/CD, centralized logging, rate-limit gateway).

8. Django deployment warnings:
- Current local settings still emit standard deployment warnings until production values are set for `DEBUG`, `SECRET_KEY`, HTTPS redirect, HSTS, and secure cookies.


## 7. Frontend UX Redesign Delivered

- Modern conversational shell with collapsible sidebar and mobile drawer
- Explicit tabs for `Chat`, `Guidance`, `Facilities`, and `Profile`
- Sticky assistant header with online status and model profile
- ChatGPT-like composer:
  - auto-growing text area
  - `Enter` to send, `Shift+Enter` for newline
  - attachment button, send state, consent toggle
- Message-level actions:
  - copy, regenerate, delete, helpful/not-helpful feedback
- Day separators and grouped timestamps for easier scanability
- Typing indicator and floating "Latest" button for long chats
- Settings modal for theme and model profile
- Light and dark modes with responsive layout for mobile and desktop

Keyboard shortcuts:

- `Ctrl+K` / `Cmd+K`: focus chat composer
- `Esc`: close settings modal and mobile sidebar

## 8. Recommended Next Actions Before Go-Live

1. Replace non-commercial image dataset/model if commercial use is required.
2. Rebalance text training distribution (especially COVID-heavy priors) to improve differential ranking stability.
3. Move DB from SQLite to managed MySQL/PostgreSQL.
4. Set production secrets/hosts/CORS and enforce HTTPS.
5. Add monitoring, alerting, and backup policy.
6. Run user acceptance test (UAT) with real clinical stakeholders.

## 9. Key Paths (Delivery Artifacts)

- Backend API: `backend/`
- Frontend app: `frontend/`
- Text model artifacts: `backend/models/`
- Dialogue model artifacts: `backend/models/`
- Image model artifacts: `backend/models/`
- Integrated text dataset outputs: `data/`
- Dialogue dataset outputs: `data/dialogue/`
- Image dataset summary: `data/image_datasets/dermamnist_summary.json`
