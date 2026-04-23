import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import django


BASE_URL = "/api/v1"


@dataclass
class CheckResult:
    name: str
    ok: bool
    status: int
    details: str = ""


def _response_details(response: Any) -> str:
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        try:
            data = response.json()
        except Exception:  # noqa: BLE001
            return response.content.decode("utf-8", errors="ignore")[:280]
        if isinstance(data, dict):
            # Keep output compact.
            snippet = {k: data[k] for k in list(data.keys())[:4]}
            return str(snippet)[:280]
        return str(data)[:280]
    return response.content.decode("utf-8", errors="ignore")[:280]


def _check(results: list[CheckResult], name: str, response: Any, expected_statuses: set[int]) -> None:
    ok = response.status_code in expected_statuses
    results.append(
        CheckResult(
            name=name,
            ok=ok,
            status=response.status_code,
            details="" if ok else _response_details(response),
        )
    )


def _print_results(results: list[CheckResult]) -> int:
    print("\nAPI smoke results")
    print("=" * 80)
    failed = [item for item in results if not item.ok]
    for item in results:
        marker = "PASS" if item.ok else "FAIL"
        print(f"[{marker}] {item.name} -> {item.status}")
        if item.details:
            print(f"       details: {item.details}")
    print("-" * 80)
    print(f"total={len(results)} pass={len(results) - len(failed)} fail={len(failed)}")
    return 1 if failed else 0


def main() -> int:
    backend_dir = Path(__file__).resolve().parents[1]
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "healthcare_ai.settings")
    django.setup()

    from django.contrib.auth.models import User
    from rest_framework.test import APIClient

    results: list[CheckResult] = []
    client = APIClient(HTTP_HOST="127.0.0.1")

    suffix = uuid.uuid4().hex[:8]
    user_email = f"smoke_{suffix}@example.com"
    user_password = "SmokeTest123"

    admin_email = f"admin_smoke_{suffix}@example.com"
    admin_username = f"admin_smoke_{suffix}"
    admin_password = "AdminSmoke123"

    # Public endpoints
    response = client.get(f"{BASE_URL}/health/")
    _check(results, "GET /health/", response, {200})

    response = client.post(
        f"{BASE_URL}/auth/register/",
        {
            "email": user_email,
            "password": user_password,
            "first_name": "Smoke",
            "last_name": "User",
            "phone_number": "0700000000",
        },
        format="json",
    )
    _check(results, "POST /auth/register/", response, {201})
    if response.status_code != 201:
        return _print_results(results)
    user_id = response.json()["user"]["id"]
    user_tokens = response.json()["tokens"]
    current_refresh = user_tokens["refresh"]
    current_access = user_tokens["access"]

    response = client.post(
        f"{BASE_URL}/auth/login/",
        {"identifier": user_email, "password": user_password},
        format="json",
    )
    _check(results, "POST /auth/login/", response, {200})
    if response.status_code == 200:
        current_access = response.json()["tokens"]["access"]
        current_refresh = response.json()["tokens"]["refresh"]

    response = client.post(
        f"{BASE_URL}/auth/refresh/",
        {"refresh": current_refresh},
        format="json",
    )
    _check(results, "POST /auth/refresh/", response, {200})
    if response.status_code == 200:
        current_access = response.json().get("access", current_access)
        current_refresh = response.json().get("refresh", current_refresh)

    # Authenticated user endpoints
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {current_access}")

    response = client.get(f"{BASE_URL}/profile/")
    _check(results, "GET /profile/", response, {200})

    response = client.patch(
        f"{BASE_URL}/profile/",
        {
            "first_name": "SmokeUpdated",
            "phone_number": "0712345678",
            "age": 29,
            "gender": "male",
            "medical_history": {"conditions": ["asthma"]},
        },
        format="json",
    )
    _check(results, "PATCH /profile/", response, {200})

    response = client.post(f"{BASE_URL}/chat/sessions/", {"title": "Smoke API Session"}, format="json")
    _check(results, "POST /chat/sessions/", response, {201})
    if response.status_code != 201:
        return _print_results(results)
    session_id = response.json()["id"]

    response = client.get(f"{BASE_URL}/chat/sessions/")
    _check(results, "GET /chat/sessions/", response, {200})

    response = client.get(f"{BASE_URL}/chat/sessions/{session_id}/messages/")
    _check(results, "GET /chat/sessions/{id}/messages/", response, {200})

    response = client.post(
        f"{BASE_URL}/chat/sessions/{session_id}/analyze/",
        {
            "symptom_text": "I have lower back pain and mild fever for several days with fatigue.",
            "consent_given": "true",
            "facility_type": "hospital",
            "search_radius_km": "5",
        },
        format="multipart",
    )
    _check(results, "POST /chat/sessions/{id}/analyze/", response, {200})

    response = client.get(f"{BASE_URL}/chat/history/")
    _check(results, "GET /chat/history/", response, {200})

    response = client.get(f"{BASE_URL}/chat/export/?format=json")
    _check(results, "GET /chat/export/?format=json", response, {200})

    response = client.get(f"{BASE_URL}/chat/export/?format=csv")
    _check(results, "GET /chat/export/?format=csv", response, {200})

    response = client.get(f"{BASE_URL}/export/profile/")
    _check(results, "GET /export/profile/", response, {200})

    response = client.post(f"{BASE_URL}/auth/logout/", {"refresh": current_refresh}, format="json")
    _check(results, "POST /auth/logout/", response, {200})

    # Public utility endpoints
    client.credentials()
    response = client.get(f"{BASE_URL}/location/emergency/")
    _check(results, "GET /location/emergency/", response, {200})

    response = client.get(f"{BASE_URL}/location/directions/?place_id=test_place_id")
    _check(results, "GET /location/directions/?place_id=...", response, {200})

    response = client.get(f"{BASE_URL}/location/nearby/?location_lat=-1.2921&location_lng=36.8219&facility_type=hospital")
    _check(results, "GET /location/nearby/", response, {200})

    response = client.post(
        f"{BASE_URL}/analyze/",
        {
            "symptom_text": "I have persistent cough and fever for three days and body ache.",
            "consent_given": "true",
        },
        format="multipart",
    )
    _check(results, "POST /analyze/", response, {200})

    # Admin endpoints
    admin_user = User.objects.filter(username=admin_username).first()
    if not admin_user:
        admin_user = User.objects.create_superuser(
            username=admin_username,
            email=admin_email,
            password=admin_password,
        )

    response = client.post(
        f"{BASE_URL}/auth/login/",
        {"identifier": admin_username, "password": admin_password},
        format="json",
    )
    _check(results, "POST /auth/login/ (admin)", response, {200})
    if response.status_code != 200:
        return _print_results(results)

    admin_access = response.json()["tokens"]["access"]
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {admin_access}")

    response = client.get(f"{BASE_URL}/admin/users/")
    _check(results, "GET /admin/users/", response, {200})

    response = client.patch(
        f"{BASE_URL}/admin/users/{user_id}/",
        {"first_name": "SmokeAdminEdited"},
        format="json",
    )
    _check(results, "PATCH /admin/users/{id}/", response, {200})

    response = client.post(
        f"{BASE_URL}/admin/facilities/",
        {
            "name": f"Smoke Facility {suffix}",
            "facility_type": "hospital",
            "specialization": "general",
            "address": "Smoke Test Road",
            "phone_number": "+254700000000",
            "latitude": -1.2921,
            "longitude": 36.8219,
            "is_emergency": False,
        },
        format="json",
    )
    _check(results, "POST /admin/facilities/", response, {201})
    if response.status_code != 201:
        return _print_results(results)
    facility_id = response.json()["id"]

    response = client.get(f"{BASE_URL}/admin/facilities/")
    _check(results, "GET /admin/facilities/", response, {200})

    response = client.patch(
        f"{BASE_URL}/admin/facilities/{facility_id}/",
        {"specialization": "family medicine"},
        format="json",
    )
    _check(results, "PATCH /admin/facilities/{id}/", response, {200})

    response = client.delete(f"{BASE_URL}/admin/facilities/{facility_id}/")
    _check(results, "DELETE /admin/facilities/{id}/", response, {204})

    response = client.get(f"{BASE_URL}/admin/analytics/")
    _check(results, "GET /admin/analytics/", response, {200})

    response = client.post(
        f"{BASE_URL}/admin/config/",
        {"action": "retrain_text_model"},
        format="json",
    )
    _check(results, "POST /admin/config/", response, {202})

    return _print_results(results)


if __name__ == "__main__":
    raise SystemExit(main())
