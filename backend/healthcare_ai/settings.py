import os
from pathlib import Path
from datetime import timedelta

# Load .env file if present (development convenience)
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())


BASE_DIR = Path(__file__).resolve().parent.parent

_secret_key_raw = os.getenv("DJANGO_SECRET_KEY", "")
if not _secret_key_raw:
    import warnings
    _secret_key_raw = "dev-only-insecure-key-change-before-production"
    warnings.warn(
        "DJANGO_SECRET_KEY is not set. Using an insecure default. "
        "Set DJANGO_SECRET_KEY in your environment before deploying.",
        stacklevel=1,
    )
SECRET_KEY = _secret_key_raw

DEBUG = os.getenv("DJANGO_DEBUG", "false").lower() == "true"

ALLOWED_HOSTS = [
    host.strip()
    for host in os.getenv("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")
    if host.strip()
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "django_rq",
    "rest_framework",
    "rest_framework_simplejwt.token_blacklist",
    "axes",
    "guidance",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "guidance.middleware.RequestLanguageMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "axes.middleware.AxesMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "healthcare_ai.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "healthcare_ai.wsgi.application"
ASGI_APPLICATION = "healthcare_ai.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": os.getenv("DB_ENGINE", "django.db.backends.postgresql"),
        "NAME": os.getenv("DB_NAME", "chat-bot"),
        "USER": os.getenv("DB_USERNAME", "postgres"),
        "PASSWORD": os.getenv("DB_PASSWORD", ""),
        "HOST": os.getenv("DB_HOST", "localhost"),
        "PORT": os.getenv("DB_PORT", "5432"),
        "OPTIONS": {
            "connect_timeout": 10,
        },
    }
}

# REQ-9: Warn loudly if SQLite is used in non-debug (production-like) mode
if not DEBUG and DATABASES["default"]["ENGINE"] == "django.db.backends.sqlite3":
    import warnings
    warnings.warn(
        "SQLite is configured in a non-DEBUG environment. "
        "Set DB_ENGINE=django.db.backends.postgresql (or mysql) and DB_NAME for production.",
        stacklevel=1,
    )

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

SECURE_SSL_REDIRECT = os.getenv("DJANGO_SECURE_SSL_REDIRECT", "false").lower() == "true"
SESSION_COOKIE_SECURE = os.getenv("DJANGO_SESSION_COOKIE_SECURE", str(not DEBUG)).lower() == "true"
CSRF_COOKIE_SECURE = os.getenv("DJANGO_CSRF_COOKIE_SECURE", str(not DEBUG)).lower() == "true"
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "same-origin"
X_FRAME_OPTIONS = "DENY"
SECURE_HSTS_SECONDS = int(os.getenv("DJANGO_HSTS_SECONDS", "0"))  # set to 31536000 in production
SECURE_HSTS_INCLUDE_SUBDOMAINS = os.getenv("DJANGO_HSTS_SUBDOMAINS", "false").lower() == "true"
SECURE_HSTS_PRELOAD = os.getenv("DJANGO_HSTS_PRELOAD", "false").lower() == "true"

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "guidance.authentication.CookieJWTAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.AnonRateThrottle",
        "rest_framework.throttling.UserRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "anon": os.getenv("THROTTLE_ANON_RATE", "60/minute"),
        "user": os.getenv("THROTTLE_USER_RATE", "120/minute"),
        "auth": os.getenv("THROTTLE_AUTH_RATE", "10/minute"),
        "analyze": os.getenv("THROTTLE_ANALYZE_RATE", "20/minute"),
    },
    # Keep "format" query parameter available for app-specific endpoints (e.g. /chat/export/?format=csv)
    # instead of treating it as DRF renderer override.
    "URL_FORMAT_OVERRIDE": None,
}

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=30),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=7),
    "ROTATE_REFRESH_TOKENS": True,
    "BLACKLIST_AFTER_ROTATION": True,
    "AUTH_HEADER_TYPES": ("Bearer",),
}

JWT_ACCESS_COOKIE_NAME = os.getenv("JWT_ACCESS_COOKIE_NAME", "healthcare_access")
JWT_REFRESH_COOKIE_NAME = os.getenv("JWT_REFRESH_COOKIE_NAME", "healthcare_refresh")
JWT_COOKIE_SECURE = os.getenv("JWT_COOKIE_SECURE", str(not DEBUG)).lower() == "true"
JWT_COOKIE_SAMESITE = os.getenv("JWT_COOKIE_SAMESITE", "Lax")
JWT_COOKIE_DOMAIN = os.getenv("JWT_COOKIE_DOMAIN") or None
JWT_COOKIE_PATH = os.getenv("JWT_COOKIE_PATH", "/")

CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:3000,http://localhost:3000",
    ).split(",")
    if origin.strip()
]
CORS_ALLOW_CREDENTIALS = True

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
GOOGLE_PLACES_RADIUS_METERS = int(os.getenv("GOOGLE_PLACES_RADIUS_METERS", "5000"))
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY", "")
GOOGLE_SEARCH_CSE_ID = os.getenv("GOOGLE_SEARCH_CSE_ID", "")

TEXT_MODEL_PATH = os.getenv("TEXT_MODEL_PATH", str(BASE_DIR / "models" / "text_classifier.joblib"))
TFIDF_VECTORIZER_PATH = os.getenv("TFIDF_VECTORIZER_PATH", str(BASE_DIR / "models" / "tfidf_vectorizer.joblib"))
TEXT_LABELS_PATH = os.getenv("TEXT_LABELS_PATH", str(BASE_DIR / "models" / "text_labels.json"))
TEXT_SVD_PATH = os.getenv("TEXT_SVD_PATH", str(BASE_DIR / "models" / "tfidf_svd.joblib"))
TRIAGE_LLM_ADAPTER_PATH = os.getenv("TRIAGE_LLM_ADAPTER_PATH", str(BASE_DIR / "models" / "triage_llm_adapter"))
TRIAGE_LLM_BASE_MODEL = os.getenv("TRIAGE_LLM_BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
USE_LLM_TRIAGE = os.getenv("USE_LLM_TRIAGE", "false").lower() in {"1", "true", "yes", "on"}
LLM_FALLBACK_TO_CLASSICAL = os.getenv("LLM_FALLBACK_TO_CLASSICAL", "true").lower() in {"1", "true", "yes", "on"}

IMAGE_MODEL_PATH = os.getenv("IMAGE_MODEL_PATH", str(BASE_DIR / "models" / "skin_cnn.keras"))
IMAGE_TORCH_MODEL_PATH = os.getenv(
    "IMAGE_TORCH_MODEL_PATH",
    str(BASE_DIR / "models" / "skin_cnn_torch.pt"),
)
IMAGE_LABELS_PATH = os.getenv("IMAGE_LABELS_PATH", str(BASE_DIR / "models" / "image_labels.json"))
IMAGE_MODEL_METADATA_PATH = os.getenv(
    "IMAGE_MODEL_METADATA_PATH",
    str(BASE_DIR / "models" / "image_model_metadata.json"),
)
IMAGE_INPUT_SIZE = int(os.getenv("IMAGE_INPUT_SIZE", "64"))

CONDITION_NAME_MAP_PATH = os.getenv(
    "CONDITION_NAME_MAP_PATH",
    str(BASE_DIR / "models" / "condition_name_map.json"),
)

CASE_RETENTION_DAYS = int(os.getenv("CASE_RETENTION_DAYS", "30"))
AUDIT_LOG_RETENTION_DAYS = int(os.getenv("AUDIT_LOG_RETENTION_DAYS", "365"))

# SSE streaming settings
SSE_POLL_INTERVAL_SECONDS = float(os.getenv("SSE_POLL_INTERVAL_SECONDS", "1.5"))
SSE_MAX_WAIT_SECONDS = float(os.getenv("SSE_MAX_WAIT_SECONDS", "620"))

# Email configuration
EMAIL_BACKEND = os.getenv("EMAIL_BACKEND", "django.core.mail.backends.console.EmailBackend")
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USE_TLS = os.getenv("EMAIL_USE_TLS", "true").lower() == "true"
EMAIL_HOST_USER = os.getenv("EMAIL_HOST_USER", "")
EMAIL_HOST_PASSWORD = os.getenv("EMAIL_HOST_PASSWORD", "")
DEFAULT_FROM_EMAIL = os.getenv("DEFAULT_FROM_EMAIL", "noreply@healthcare-ai.local")
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "http://127.0.0.1:5173")

DIALOGUE_INTENT_MODEL_PATH = os.getenv(
    "DIALOGUE_INTENT_MODEL_PATH",
    str(BASE_DIR / "models" / "dialogue_intent_classifier.joblib"),
)
DIALOGUE_INTENT_VECTORIZER_PATH = os.getenv(
    "DIALOGUE_INTENT_VECTORIZER_PATH",
    str(BASE_DIR / "models" / "dialogue_intent_vectorizer.joblib"),
)
DIALOGUE_RESPONSE_TEMPLATES_PATH = os.getenv(
    "DIALOGUE_RESPONSE_TEMPLATES_PATH",
    str(BASE_DIR / "models" / "dialogue_response_templates.json"),
)

AUTHENTICATION_BACKENDS = [
    "axes.backends.AxesStandaloneBackend",
    "django.contrib.auth.backends.ModelBackend",
]

# django-axes: brute-force / rate-limit on login
AXES_FAILURE_LIMIT = int(os.getenv("AXES_FAILURE_LIMIT", "5"))
AXES_COOLOFF_TIME = float(os.getenv("AXES_COOLOFF_MINUTES", "15")) / 60  # hours
AXES_LOCKOUT_PARAMETERS = ["username", "ip_address"]
AXES_RESET_ON_SUCCESS = True
AXES_ENABLE_ADMIN = True

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/1")
RQ_ANALYSIS_QUEUE = os.getenv("RQ_ANALYSIS_QUEUE", "analysis")
RQ_JOB_TIMEOUT = int(os.getenv("RQ_JOB_TIMEOUT", "600"))
RQ_QUEUES = {
    RQ_ANALYSIS_QUEUE: {
        "URL": REDIS_URL,
        "DEFAULT_TIMEOUT": RQ_JOB_TIMEOUT,
    },
}

# Django cache — use Redis when available, fall back to local memory
_redis_url = REDIS_URL
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": _redis_url,
        "OPTIONS": {
            "socket_connect_timeout": 2,
            "socket_timeout": 2,
        },
        "KEY_PREFIX": "healthcare",
    }
}

# Google Places API result cache TTL (1 hour)
PLACES_CACHE_TTL_SECONDS = int(os.getenv("PLACES_CACHE_TTL_SECONDS", "3600"))
