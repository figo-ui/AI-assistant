# AI Healthcare Chatbot System

**Advisor:** Mr. Alemu Gudeta | **Contact:** 0919778608

A full-stack, bilingual AI-powered healthcare assistant that uses machine learning to analyze symptoms, predict possible conditions, score risk levels, and guide users to the nearest healthcare facility.

---

## Quick Start

### Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.10+ |
| PostgreSQL | 14+ |
| Node.js | 18+ (only if using the React frontend) |
| Redis | 6+ (optional — needed for async jobs and caching) |

### 1 — Clone and set up the backend

```bash
git clone https://github.com/figo-ui/AI-assistant.git
cd AI-assistant/backend

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2 — Configure environment

```bash
copy .env.example .env      # Windows
cp .env.example .env        # macOS/Linux
```

Open `.env` and set at minimum:

```env
DJANGO_SECRET_KEY=your-50-char-random-secret
DB_NAME=chat-bot
DB_USERNAME=postgres
DB_PASSWORD=your-postgres-password
GOOGLE_MAPS_API_KEY=AIzaSyDXDoa_VBI5h9KuoTUs-rQF8Ve4qHxWu5M
```

### 3 — Set up the database

Make sure PostgreSQL is running and the database `chat-bot` exists, then:

```bash
python manage.py migrate
python manage.py seed_admin --username admin --email admin@example.com --password Admin1234
```

### 4 — Run the server

```bash
python manage.py runserver
```

Open your browser at **http://127.0.0.1:8000** — the plain HTML/CSS/JS frontend is served directly by Django.

### 5 — (Optional) Start the async worker

Required only for background analysis jobs:

```bash
python scripts/run_rq_worker.py
```

---

## Project Structure

```
AI-assistant/
├── backend/                        Django REST API
│   ├── guidance/                   Main app
│   │   ├── models.py               Database models
│   │   ├── views.py                API endpoints
│   │   ├── urls.py                 URL routing
│   │   ├── serializers.py          Request/response validation
│   │   └── services/               ML pipeline services
│   │       ├── pipeline.py         Main inference orchestrator
│   │       ├── text_model.py       Symptom → condition classifier
│   │       ├── image_model.py      Skin CNN inference
│   │       ├── fusion.py           Text + image fusion engine
│   │       ├── risk.py             Risk scoring engine
│   │       ├── clinical_safety.py  Emergency pattern overrides
│   │       ├── facilities.py       Google Places + local DB lookup
│   │       ├── language_support.py Bilingual EN/Amharic support
│   │       └── pii_redaction.py    PHI redaction (Presidio)
│   ├── models/                     Trained ML artifacts
│   │   ├── text_classifier.joblib
│   │   ├── tfidf_vectorizer.joblib
│   │   ├── text_labels.json
│   │   ├── dialogue_intent_classifier.joblib
│   │   └── skin_cnn_torch.pt
│   ├── scripts/                    Training and utility scripts
│   │   ├── retrain_all_models.py   One-shot retraining
│   │   └── quick_retrain.py        Fast text model retrain
│   └── requirements.txt
│
├── frontend_html/                  Plain HTML/CSS/JS frontend
│   ├── index.html                  Single entry point
│   ├── css/                        Stylesheets
│   └── js/
│       ├── app.js                  Main controller
│       ├── api/                    API client modules
│       ├── components/             UI components
│       └── pages/                  Page renderers
│
├── data/                           Training datasets (not committed)
├── docker-compose.yml
└── README.md
```

---

## API Reference

Base URL: `http://localhost:8000/api/v1/`

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/health/` | None | Health check |
| POST | `/auth/register/` | None | Register new user |
| POST | `/auth/login/` | None | Login |
| POST | `/auth/refresh/` | Cookie | Refresh JWT tokens |
| POST | `/auth/logout/` | JWT | Logout |
| GET/PATCH | `/profile/` | JWT | User profile |
| GET/POST | `/chat/sessions/` | JWT | Chat sessions |
| POST | `/chat/sessions/{id}/analyze/` | JWT | Symptom analysis |
| GET | `/location/nearby/` | None | Find nearby facilities |
| GET | `/location/emergency/` | None | Emergency contacts |
| GET | `/admin/analytics/` | Staff | Platform analytics |

---

## Retraining the Models

### Text model (CPU, ~5 minutes)

```bash
cd backend
python scripts/retrain_all_models.py --text-only
```

Uses `data/unified/ULTIMATE_TRIAGE_KNOWLEDGE.csv` (43,621 rows, 202 conditions).

### All models

```bash
python scripts/retrain_all_models.py
```

### After retraining — run regression tests

```bash
python scripts/run_triage_regression.py
```

Expected: 40/40 top-3 pass, 40/40 risk pass, 20/20 emergency flag.

---

## Docker

```bash
cp backend/.env.example backend/.env   # fill in values
docker-compose up --build
```

Services: backend (8000), postgres (5432), redis (6379)

---

## License

See `CLIENT_HANDOFF_PACK.md` for dataset license notes.
