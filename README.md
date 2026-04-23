# AI Healthcare Assistant

A full-stack multilingual healthcare assistant chatbot with symptom triage, image analysis, and clinical guidance.

**Stack:** Django 5 · React 18 · PostgreSQL · Redis · PyTorch · scikit-learn

---

## Quick Start

### Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.10+ |
| Node.js | 18+ |
| PostgreSQL | 14+ |
| Redis | 6+ |

---

### 1 — Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# Copy and fill in environment variables
copy .env.example .env      # Windows
cp .env.example .env        # macOS/Linux

python manage.py migrate
python manage.py seed_admin --username admin --email admin@example.com --password Admin1234
python manage.py runserver
```

Backend runs at → `http://127.0.0.1:8000`

Start the async worker in a second terminal:

```bash
python scripts/run_rq_worker.py
```

---

### 2 — Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at → `http://127.0.0.1:5173`

Production build:

```bash
npm run build
npm run preview
```

---

### 3 — Docker (full stack)

```bash
cp backend/.env.example backend/.env   # fill in values
docker-compose up --build
```

Services:
- `backend` → `http://localhost:8000`
- `frontend` → `http://localhost:5173`
- `postgres` → port 5432
- `redis` → port 6379

---

## Environment Variables

Copy `backend/.env.example` to `backend/.env` and set:

| Variable | Description |
|----------|-------------|
| `DJANGO_SECRET_KEY` | 50+ char random string (required in production) |
| `DJANGO_DEBUG` | `false` in production |
| `DJANGO_ALLOWED_HOSTS` | Comma-separated hostnames |
| `DB_NAME`, `DB_USERNAME`, `DB_PASSWORD` | PostgreSQL credentials |
| `REDIS_URL` | Redis connection string |
| `GOOGLE_MAPS_API_KEY` | Optional — enables live facility search |
| `EMAIL_HOST_USER`, `EMAIL_HOST_PASSWORD` | SMTP credentials for email verification |

See `backend/.env.example` for the full list.

---

## API Reference

Base path: `http://localhost:8000/api/v1/`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health/` | Health check |
| POST | `/auth/register/` | Register user |
| POST | `/auth/login/` | Login |
| POST | `/auth/refresh/` | Refresh JWT |
| POST | `/auth/logout/` | Logout |
| GET/PATCH | `/profile/` | User profile |
| GET/POST | `/chat/sessions/` | Chat sessions |
| POST | `/chat/sessions/{id}/analyze/` | Symptom analysis (text + image) |
| GET | `/chat/history/` | Chat history |
| GET | `/chat/export/` | Export history (JSON/CSV) |
| POST | `/analyze/` | Public anonymous analysis |
| GET | `/location/nearby/` | Nearby facilities |
| GET | `/location/emergency/` | Emergency contacts |
| GET | `/admin/analytics/` | Admin analytics |

Full API documentation: see `CLIENT_HANDOFF_PACK.md`

---

## Model Artifacts

Place trained model files in `backend/models/`:

| File | Description |
|------|-------------|
| `text_classifier.joblib` | Symptom → condition classifier (113 classes) |
| `tfidf_vectorizer.joblib` | TF-IDF vectorizer |
| `text_labels.json` | Label index |
| `dialogue_intent_classifier.joblib` | Dialogue intent classifier |
| `dialogue_intent_vectorizer.joblib` | Dialogue vectorizer |
| `dialogue_response_templates.json` | Response templates |
| `skin_cnn_torch.pt` | Dermatology CNN (PyTorch) |
| `image_labels.json` | Image class labels |

> Model binaries are excluded from git. Download from the shared model registry or retrain using `backend/scripts/`.

---

## Project Structure

```
├── backend/                  # Django REST API
│   ├── guidance/             # Main app (models, views, services)
│   │   └── services/         # ML pipeline, triage, RAG, safety
│   ├── healthcare_ai/        # Django settings & routing
│   ├── models/               # Trained model artifacts (gitignored)
│   ├── scripts/              # Training & utility scripts
│   └── requirements.txt
├── frontend/                 # React + TypeScript UI
│   └── src/
│       └── features/         # auth, chat, facilities, profile, admin
├── deployment_package/       # Docker inference service
├── CLIENT_HANDOFF_PACK.md    # Full delivery documentation
└── docker-compose.yml
```

---

## Known Limitations

- Image model (DermaMNIST) is prototype-grade (F1: 0.31) — not for clinical use
- System output is decision support only, not a clinical diagnosis
- Commercial deployment requires dataset license review (see `CLIENT_HANDOFF_PACK.md §6`)
- Production hardening (TLS, reverse proxy, CI/CD) is not included

---

## License

See `CLIENT_HANDOFF_PACK.md` for dataset license notes and usage restrictions.
