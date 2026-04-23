# AI Healthcare Assistant Backend

This backend is a Django monolith for a multilingual healthcare-assistant prototype with:

- HttpOnly cookie auth (`register/login/refresh/logout`)
- profile + chat session management
- text triage, optional image branch, fusion, and rule-based safety overrides
- English/Amharic language detection and localized responses
- consent-gated real-time search with PHI redaction before external calls
- anonymous async polling secured by a per-case UUID status token
- RQ + Redis background execution for async analysis

Important: this is clinical decision support only, not autonomous diagnosis.

## Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your values (PostgreSQL credentials are pre-configured):

```bash
copy .env.example .env
```

The default `.env` connects to the local PostgreSQL database `chat-bot` on port 5432.
Make sure PostgreSQL is running and the database exists in pgAdmin before migrating.

Run Redis locally before using async analysis:

```bash
redis-server
```

Apply migrations and start Django:

```bash
python manage.py migrate
python manage.py runserver
```

Seed the admin user (first run only):

```bash
python manage.py seed_admin --username admin --email admin@example.com --password Admin1234
```

Start the RQ worker in a second shell:

```bash
python scripts/run_rq_worker.py
```

## Auth Model

- JWT access/refresh tokens are issued into HttpOnly cookies.
- Frontend requests must send `credentials: "include"`.
- Refresh uses `POST /api/v1/auth/refresh/` and rotates cookies.
- Logout clears both auth cookies.

## API Notes

- Health: `GET /api/v1/health/`
- Public analyze: `POST /api/v1/analyze/`
- Public async status: `GET /api/v1/analyze/{case_id}/?token=<uuid>`
- Session analyze: `POST /api/v1/chat/sessions/{session_id}/analyze/`
- Search is only attempted when:
  - the query looks freshness-sensitive, and
  - `search_consent_given=true`

Before external search, symptom text is redacted with Presidio + regex fallbacks so only sanitized text is sent to PubMed or DuckDuckGo.

## Key Runtime Features

### Bilingual behavior

- `langdetect` + Amharic-script heuristics determine response language
- Amharic symptom phrases are normalized into English clinical keywords for local models
- risk labels, disclaimers, and recommendations are localized

### Async analysis

- Async requests are queued through Redis/RQ
- Anonymous polling requires the returned `status_token`
- Authenticated users can poll their own cases without the token

### Search routing

- current/freshness prompts such as `latest`, `current`, `guideline`, `outbreak`, `2026`, `Ethiopia`, `malaria`
- PubMed first, DuckDuckGo fallback
- search results are attached under `search_context`

## Model Artifacts

Expected under `backend/models/`:

- `text_classifier.joblib`
- `tfidf_vectorizer.joblib`
- `text_labels.json`
- `tfidf_svd.joblib` (optional)
- `dialogue_intent_classifier.joblib`
- `dialogue_intent_vectorizer.joblib`
- `dialogue_response_templates.json`
- `skin_cnn_torch.pt` or `skin_cnn.keras`
- `image_labels.json`
- `triage_llm_adapter/` (optional LoRA adapter)

If artifacts are missing, the backend falls back to safer demo behavior where possible.

## Training

### Text model

```bash
python scripts/preprocess_and_train.py --input "..\data\expanded_symptom_condition.csv" --min-samples 5 --model-out-dir "models"
python scripts/evaluate_text_model.py --dataset "..\data\synthea_symptom_condition.csv"
```

### LLM dataset

The supervised dataset now emits strict JSON assistant targets matching the runtime `llm_triage.py` schema:

```bash
python scripts/prepare_triage_llm_dataset.py --input-csv "..\data\expanded_symptom_condition_clean_processed.csv"
```

### Image model

Local retraining is CPU-heavy; prefer Kaggle for real training. Local command:

```bash
python scripts/train_image_model.py --manifest "data\\image_dataset_combined\\manifest.jsonl" --architecture efficientnet_b3 --epochs 8
```

## Kaggle

Use the checked-in workflow documents:

- `backend/notebooks/kaggle_llm_qlora_workflow.md`
- `backend/notebooks/kaggle_image_training_workflow.md`

Those notebooks are intended for free Kaggle GPUs (`T4` or `P100`) because the local machine is not suitable for 7B QLoRA or modern vision fine-tuning.

## Regression + Retention

Refresh the regression report:

```bash
python scripts/run_triage_regression.py
```

Purge expired PHI-bearing records:

```bash
python scripts/purge_phi_data.py --case-retention-days 30 --audit-retention-days 365
```
