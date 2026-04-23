FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source
COPY . .

# Static files dir
RUN mkdir -p /app/media /app/staticfiles

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
