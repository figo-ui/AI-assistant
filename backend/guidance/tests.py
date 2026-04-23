from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from .models import CaseSubmission, HealthcareFacility, InferenceRecord, RiskAssessment
from .serializers import AnalyzeCaseSerializer
from .services.clinical_safety import apply_prediction_safety_overrides, build_safety_summary
from .services.facilities import lookup_nearby_facilities
from .services.language_support import translate_dynamic_text
from .services.llm_triage import _extract_json_payload
from .services.retention import purge_expired_phi_records
from .services.search_router import run_search_router
from .services.schema import validate_triage_response


class AnalyzeCaseSerializerTests(TestCase):
    def test_requires_consent(self):
        serializer = AnalyzeCaseSerializer(
            data={
                "symptom_text": "persistent chest pain with sweating and shortness of breath",
                "consent_given": False,
            }
        )

        self.assertFalse(serializer.is_valid())
        self.assertIn("consent_given", serializer.errors)

    def test_requires_both_location_coordinates(self):
        serializer = AnalyzeCaseSerializer(
            data={
                "symptom_text": "persistent lower back pain with fever for several days",
                "consent_given": True,
                "location_lat": 1.2345,
            }
        )

        self.assertFalse(serializer.is_valid())
        self.assertIn("Both location_lat and location_lng are required", str(serializer.errors))

    def test_accepts_async_mode(self):
        serializer = AnalyzeCaseSerializer(
            data={
                "symptom_text": "fever with cough and chest discomfort for three days",
                "consent_given": True,
                "async_mode": True,
            }
        )

        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertTrue(serializer.validated_data["async_mode"])

    def test_search_consent_defaults_false(self):
        serializer = AnalyzeCaseSerializer(
            data={
                "symptom_text": "latest malaria update with fever and chills",
                "consent_given": True,
                "force_search": True,
            }
        )

        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertFalse(serializer.validated_data["search_consent_given"])


class ClinicalSafetyTests(TestCase):
    def test_stroke_pattern_promotes_emergency_condition(self):
        predictions = [
            {"condition": "Migraine", "probability": 0.7},
            {"condition": "Stress response", "probability": 0.2},
        ]

        adjusted = apply_prediction_safety_overrides(
            "severe headache with one side weakness and slurred speech",
            predictions,
            top_k=5,
        )

        self.assertTrue(adjusted)
        self.assertIn(adjusted[0]["condition"], {"Stroke", "Paralysis (brain hemorrhage)"})
        self.assertGreaterEqual(adjusted[0]["probability"], 0.3)

    def test_anaphylaxis_summary_escalates_to_high_risk(self):
        summary = build_safety_summary(
            "allergy after food with lip swelling throat swelling and trouble breathing"
        )

        self.assertEqual(summary["risk_level"], "High")
        self.assertIn("possible anaphylaxis pattern", summary["red_flags"])


class FacilityLookupTests(TestCase):
    @override_settings(GOOGLE_MAPS_API_KEY="")
    def test_local_facility_lookup_returns_nearby_result(self):
        HealthcareFacility.objects.create(
            name="Downtown Clinic",
            facility_type=HealthcareFacility.FacilityType.CLINIC,
            specialization="general practice",
            address="1 Main Street",
            phone_number="555-0101",
            latitude=1.0,
            longitude=36.0,
        )

        results = lookup_nearby_facilities(
            location_lat=1.001,
            location_lng=36.001,
            facility_type="clinic",
            specialization="general",
            radius_km=5,
            limit=5,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["provider_name"], "Downtown Clinic")
        self.assertEqual(results[0]["source"], "local_registry")

    def test_emergency_contacts_are_locale_aware(self):
        client = APIClient()

        response = client.get("/api/v1/location/emergency/?country_code=KE")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["contacts"][0]["region"], "Kenya")


class HealthViewTests(TestCase):
    def test_health_endpoint_is_public(self):
        client = APIClient()

        response = client.get("/api/v1/health/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


class AnalyzeCaseViewTests(TestCase):
    @patch("guidance.views.submit_async_case_analysis")
    def test_public_analyze_supports_async_mode(self, submit_async_case_analysis_mock):
        client = APIClient()

        response = client.post(
            "/api/v1/analyze/",
            {
                "symptom_text": "fever, cough, and shortness of breath for two days",
                "consent_given": True,
                "async_mode": True,
            },
        )

        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertEqual(payload["status"], "queued")
        self.assertIn("case_id", payload)
        submit_async_case_analysis_mock.assert_called_once_with(payload["case_id"])

    def test_completed_analysis_status_returns_serialized_result(self):
        user = User.objects.create_user(username="status-user", password="Secret123A")
        case = CaseSubmission.objects.create(
            user=user,
            symptom_text="sudden chest pain and sweating",
            consent_given=True,
            status="completed",
            metadata={"language": "en"},
        )
        InferenceRecord.objects.create(
            case=case,
            fused_predictions=[{"condition": "Heart attack", "probability": 0.88}],
            text_predictions=[{"condition": "Heart attack", "probability": 0.88}],
            image_predictions=[],
            text_confidence=0.88,
            image_confidence=0.0,
            fusion_confidence=0.88,
        )
        RiskAssessment.objects.create(
            case=case,
            risk_score=0.91,
            risk_level="High",
            recommendation_text="Seek urgent in-person care now.",
            disclaimer_text="This is not medical advice.",
            needs_urgent_care=True,
        )

        client = APIClient()
        client.force_authenticate(user=user)

        response = client.get(f"/api/v1/analyze/{case.id}/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["result"]["probable_conditions"][0]["condition"], "Heart attack")
        self.assertEqual(payload["result"]["risk_level"], "High")

    @patch("guidance.views.submit_async_case_analysis")
    def test_anonymous_async_status_requires_secret_token(self, submit_async_case_analysis_mock):
        client = APIClient()

        response = client.post(
            "/api/v1/analyze/",
            {
                "symptom_text": "latest malaria Ethiopia 2026 with fever and chills",
                "consent_given": True,
                "async_mode": True,
            },
        )

        self.assertEqual(response.status_code, 202)
        payload = response.json()
        case_id = payload["case_id"]
        token = payload["status_token"]
        self.assertTrue(token)

        denied = client.get(f"/api/v1/analyze/{case_id}/")
        self.assertEqual(denied.status_code, 404)

        allowed = client.get(f"/api/v1/analyze/{case_id}/?token={token}")
        self.assertEqual(allowed.status_code, 200)
        submit_async_case_analysis_mock.assert_called_once_with(case_id)


class AuthCookieTests(TestCase):
    def test_login_sets_http_only_auth_cookies(self):
        user = User.objects.create_user(username="cookie-user", email="cookie@example.com", password="Secret123A")
        client = APIClient()

        response = client.post(
            "/api/v1/auth/login/",
            {"identifier": user.username, "password": "Secret123A"},
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("healthcare_access", response.cookies)
        self.assertIn("healthcare_refresh", response.cookies)
        self.assertTrue(response.cookies["healthcare_access"]["httponly"])


class SearchRouterTests(TestCase):
    def test_search_router_blocks_external_lookup_without_consent(self):
        payload = run_search_router(
            "latest malaria Ethiopia 2026 fever",
            translated_query="latest malaria Ethiopia 2026 fever",
            force_search=True,
            search_consent=False,
        )

        self.assertTrue(payload["blocked_by_policy"])
        self.assertEqual(payload["results"], [])


class LocalizationTests(TestCase):
    def test_translate_dynamic_text_preserves_specific_recommendation(self):
        translated = translate_dynamic_text(
            "Possible stroke pattern: call emergency services or go to the emergency department immediately.",
            "am",
        )

        self.assertIn("ስትሮክ", translated)


class StructuredLLMTests(TestCase):
    def test_extract_json_payload_ignores_wrapping_text(self):
        payload = _extract_json_payload(
            'assistant:\n{"predictions":[{"condition":"Stroke","probability":0.6}],"risk_level":"high","red_flags":["slurred speech"],"recommendation":"Go now","reasoning":"Acute neurologic deficits"}'
        )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["risk_level"], "high")

    def test_validate_triage_response_normalizes_probabilities_and_risk(self):
        response = validate_triage_response(
            {
                "predictions": [
                    {"condition": "Stroke", "probability": 4},
                    {"condition": "Migraine", "probability": 1},
                ],
                "risk_level": "HIGH",
                "red_flags": ["slurred speech"],
                "recommendation": "Seek urgent care now",
                "reasoning": "Focal neurologic symptoms",
            }
        )

        self.assertEqual(response.risk_level, "High")
        self.assertAlmostEqual(
            sum(item.probability for item in response.predictions),
            1.0,
            places=3,
        )
        self.assertEqual(response.predictions[0].condition, "Stroke")


class RetentionTests(TestCase):
    def test_purge_expired_phi_records_respects_retention_window(self):
        old_case = CaseSubmission.objects.create(
            symptom_text="old case",
            consent_given=True,
            status="completed",
        )
        recent_case = CaseSubmission.objects.create(
            symptom_text="recent case",
            consent_given=True,
            status="completed",
        )
        CaseSubmission.objects.filter(id=old_case.id).update(created_at="2024-01-01T00:00:00Z")

        summary = purge_expired_phi_records(case_retention_days=30, now="2026-03-08T00:00:00Z")

        self.assertEqual(summary["cases_deleted"], 1)
        self.assertFalse(CaseSubmission.objects.filter(id=old_case.id).exists())
        self.assertTrue(CaseSubmission.objects.filter(id=recent_case.id).exists())
