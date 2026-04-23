import csv
import io
import json
import logging
import re
import time
from typing import Optional

from django.conf import settings
from django.contrib.auth import authenticate, login as auth_login
from django.contrib.auth.models import User
from django.db import transaction
from django.db.models import Count, Q
from django.http import HttpResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404
from django.utils.crypto import constant_time_compare
from django.utils.dateparse import parse_date
from django.utils import timezone
from rest_framework import permissions, status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken

from .models import (
    AuditLog,
    CaseSubmission,
    ChatMessage,
    ChatSession,
    HealthcareFacility,
    UserProfile,
)
from .serializers import (
    AdminUserUpdateSerializer,
    AnalyzeCaseSerializer,
    ChatMessageSerializer,
    ChatSessionSerializer,
    FacilitySearchSerializer,
    HealthcareFacilitySerializer,
    LoginSerializer,
    ProfileUpdateSerializer,
    RegisterSerializer,
    UserProfileSerializer,
    UserSerializer,
)
from .services.facilities import emergency_contacts, lookup_nearby_facilities
from .services.async_inference import async_case_status, submit_async_case_analysis
from .services.pipeline import run_case_analysis
from .services.dialogue_style import build_supportive_opening
from .services.language_support import build_assistant_summary, detect_language
from .services.pii_redaction import redact_phi_text
from .services.email_service import (
    send_emergency_alert_email,
    send_emergency_contact_alert,
    send_verification_email,
)
from .throttles import AnalyzeAnonRateThrottle, AnalyzeRateThrottle, AuthRateThrottle
import django_rq

logger = logging.getLogger(__name__)
GENERIC_CONDITION_RE = re.compile(r"^(condition\s+\d+|class_\d+)$", re.IGNORECASE)


def _tokens_for_user(user: User):
    refresh = RefreshToken.for_user(user)
    return {"refresh": str(refresh), "access": str(refresh.access_token)}


def _set_auth_cookies(response: Response, *, refresh_token: str, access_token: str) -> Response:
    common = {
        "httponly": True,
        "secure": bool(getattr(settings, "JWT_COOKIE_SECURE", True)),
        "samesite": getattr(settings, "JWT_COOKIE_SAMESITE", "Lax"),
        "domain": getattr(settings, "JWT_COOKIE_DOMAIN", None),
        "path": getattr(settings, "JWT_COOKIE_PATH", "/"),
    }
    response.set_cookie(
        getattr(settings, "JWT_ACCESS_COOKIE_NAME", "healthcare_access"),
        access_token,
        max_age=int(settings.SIMPLE_JWT["ACCESS_TOKEN_LIFETIME"].total_seconds()),
        **common,
    )
    response.set_cookie(
        getattr(settings, "JWT_REFRESH_COOKIE_NAME", "healthcare_refresh"),
        refresh_token,
        max_age=int(settings.SIMPLE_JWT["REFRESH_TOKEN_LIFETIME"].total_seconds()),
        **common,
    )
    return response


def _clear_auth_cookies(response: Response) -> Response:
    response.delete_cookie(
        getattr(settings, "JWT_ACCESS_COOKIE_NAME", "healthcare_access"),
        path=getattr(settings, "JWT_COOKIE_PATH", "/"),
        domain=getattr(settings, "JWT_COOKIE_DOMAIN", None),
        samesite=getattr(settings, "JWT_COOKIE_SAMESITE", "Lax"),
    )
    response.delete_cookie(
        getattr(settings, "JWT_REFRESH_COOKIE_NAME", "healthcare_refresh"),
        path=getattr(settings, "JWT_COOKIE_PATH", "/"),
        domain=getattr(settings, "JWT_COOKIE_DOMAIN", None),
        samesite=getattr(settings, "JWT_COOKIE_SAMESITE", "Lax"),
    )
    return response


def _resolve_user_from_identifier(identifier: str) -> Optional[User]:
    identifier = (identifier or "").strip()
    if not identifier:
        return None
    if "@" in identifier:
        return User.objects.filter(email__iexact=identifier).first()
    return User.objects.filter(username__iexact=identifier).first()


def _session_for_user(user: User, session_id: int) -> ChatSession:
    return get_object_or_404(ChatSession, id=session_id, user=user)


def _profile_for_user(user: User) -> UserProfile:
    profile, _ = UserProfile.objects.get_or_create(user=user)
    return profile


def _profile_medical_payload(profile: UserProfile) -> dict:
    if profile.medical_profile:
        return dict(profile.medical_profile)
    if profile.medical_history:
        return dict(profile.medical_history)
    return {}


def _assistant_summary(result: dict, symptom_text: str = "", language: str = "en") -> str:
    if language == "am" or (result.get("search_context") or {}).get("results"):
        return build_assistant_summary(result, language)

    conditions = result.get("probable_conditions", [])[:3]
    named_conditions = [
        item
        for item in conditions
        if not GENERIC_CONDITION_RE.fullmatch(str(item.get("condition", "")).strip())
        and str(item.get("condition", "")).strip().lower() != "unspecified clinical pattern"
    ]

    if named_conditions:
        cond_summary = ", ".join(
            f"{item.get('condition', 'Unknown')} ({float(item.get('probability', 0)) * 100:.1f}%)"
            for item in named_conditions[:3]
        )
        clinical_summary = f"Possible causes from your symptoms: {cond_summary}. "
    else:
        clinical_summary = (
            "I could not identify a specific named condition with strong confidence from the current input. "
        )

    supportive_opening = build_supportive_opening(
        symptom_text=symptom_text,
        risk_level=str(result.get("risk_level", "low")),
    )

    return (
        f"{supportive_opening} "
        f"{clinical_summary}"
        f"Current risk level is {result.get('risk_level', 'Unknown')} "
        f"(score {float(result.get('risk_score', 0)):.2f}). "
        f"Recommended next step: {result.get('recommendation_text', '')}\n\n"
        f"*Disclaimer: Image findings utilize an experimental diagnostic model with limited reliability. Do not substitute for professional assessment.*"
    ).strip()


def _serialize_case_result(case: CaseSubmission) -> dict:
    inference = getattr(case, "inference", None)
    risk = getattr(case, "risk", None)
    fused_predictions = inference.fused_predictions if inference else []
    user_facing = [
        item
        for item in fused_predictions
        if not GENERIC_CONDITION_RE.fullmatch(str(item.get("condition", "")).strip())
    ]
    if not user_facing and fused_predictions:
        top_item = dict(fused_predictions[0])
        top_item["condition"] = "Unspecified clinical pattern"
        user_facing = [top_item]
    return {
        "case_id": case.id,
        "created_at": case.created_at.isoformat(),
        "symptom_text": case.symptom_text,
        "symptom_tags": case.symptom_tags,
        "status": case.status,
        "response_language": (case.metadata or {}).get("language", "en"),
        "probable_conditions": user_facing,
        "raw_probable_conditions": fused_predictions,
        "risk_level": risk.risk_level if risk else None,
        "risk_score": risk.risk_score if risk else None,
        "recommendation_text": risk.recommendation_text if risk else "",
        "red_flags": risk.red_flags if risk else [],
        "prevention_advice": risk.prevention_advice if risk else [],
    }


class HealthView(APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def get(self, request):
        return Response({"status": "ok"}, status=status.HTTP_200_OK)


class RegisterView(APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    parser_classes = [JSONParser, FormParser, MultiPartParser]
    throttle_classes = []  # uses AuthRateThrottle via settings scope

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()

        # REQ-1: Create email verification token and send verification email
        from .models import EmailVerificationToken
        ev_token, _ = EmailVerificationToken.objects.get_or_create(user=user)
        send_verification_email(user, str(ev_token.token))

        tokens = _tokens_for_user(user)
        response = Response(
            {
                "user": UserSerializer(user).data,
                "profile": UserProfileSerializer(_profile_for_user(user)).data,
                "email_verification_sent": bool(user.email),
            },
            status=status.HTTP_201_CREATED,
        )
        return _set_auth_cookies(
            response,
            refresh_token=tokens["refresh"],
            access_token=tokens["access"],
        )


class LoginView(APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    parser_classes = [JSONParser, FormParser, MultiPartParser]
    throttle_classes = [AuthRateThrottle]

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        identifier = serializer.validated_data["identifier"]
        password = serializer.validated_data["password"]

        user_obj = _resolve_user_from_identifier(identifier)
        username = user_obj.username if user_obj else identifier

        # axes lockout check happens inside authenticate via AxesStandaloneBackend
        user = authenticate(request=request, username=username, password=password)

        if user is None:
            # Record failed attempt for axes
            from axes.handlers.proxy import AxesProxyHandler
            AxesProxyHandler.user_login_failed(
                sender=self.__class__,
                credentials={"username": username},
                request=request,
            )
            return Response({"error": "Invalid credentials."}, status=status.HTTP_401_UNAUTHORIZED)
        if not user.is_active:
            return Response({"error": "Account is disabled."}, status=status.HTTP_403_FORBIDDEN)

        # Signal successful login to axes so it resets failure count
        auth_login(request, user, backend="axes.backends.AxesStandaloneBackend")

        tokens = _tokens_for_user(user)
        response = Response(
            {
                "user": UserSerializer(user).data,
                "profile": UserProfileSerializer(_profile_for_user(user)).data,
            },
            status=status.HTTP_200_OK,
        )
        return _set_auth_cookies(
            response,
            refresh_token=tokens["refresh"],
            access_token=tokens["access"],
        )


class CookieTokenRefreshView(APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    parser_classes = [JSONParser]

    def post(self, request):
        refresh_token = (
            request.data.get("refresh")
            or request.COOKIES.get(getattr(settings, "JWT_REFRESH_COOKIE_NAME", "healthcare_refresh"))
        )
        if not refresh_token:
            response = Response({"error": "Refresh token missing."}, status=status.HTTP_401_UNAUTHORIZED)
            return _clear_auth_cookies(response)
        try:
            refresh = RefreshToken(refresh_token)
        except Exception:
            response = Response({"error": "Invalid refresh token."}, status=status.HTTP_401_UNAUTHORIZED)
            return _clear_auth_cookies(response)

        if settings.SIMPLE_JWT.get("BLACKLIST_AFTER_ROTATION", False):
            try:
                refresh.blacklist()
            except Exception:
                pass

        new_refresh = RefreshToken.for_user(User.objects.get(id=refresh["user_id"]))
        response = Response({"status": "refreshed"}, status=status.HTTP_200_OK)
        return _set_auth_cookies(
            response,
            refresh_token=str(new_refresh),
            access_token=str(new_refresh.access_token),
        )


class LogoutView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [JSONParser]

    def post(self, request):
        refresh_token = (
            request.data.get("refresh")
            or request.COOKIES.get(getattr(settings, "JWT_REFRESH_COOKIE_NAME", "healthcare_refresh"))
        )
        if refresh_token:
            try:
                token = RefreshToken(refresh_token)
                token.blacklist()
            except Exception:
                response = Response({"error": "Invalid refresh token."}, status=status.HTTP_400_BAD_REQUEST)
                return _clear_auth_cookies(response)
        response = Response({"status": "logged_out"}, status=status.HTTP_200_OK)
        return _clear_auth_cookies(response)


class ProfileView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [JSONParser, FormParser, MultiPartParser]

    def get(self, request):
        profile = _profile_for_user(request.user)
        return Response(UserProfileSerializer(profile).data, status=status.HTTP_200_OK)

    def patch(self, request):
        serializer = ProfileUpdateSerializer(data=request.data, context={"request": request}, partial=True)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        user = request.user
        profile = _profile_for_user(user)
        user_dirty_fields = []
        profile_dirty_fields = []

        for field in ("first_name", "last_name", "email"):
            if field in data:
                setattr(user, field, data[field])
                user_dirty_fields.append(field)
        if user_dirty_fields:
            user.save(update_fields=user_dirty_fields)

        for field in (
            "phone_number",
            "age",
            "gender",
            "address",
            "emergency_contact_name",
            "emergency_contact_phone",
            "preferred_language",
        ):
            if field in data:
                setattr(profile, field, data[field])
                profile_dirty_fields.append(field)
        if "medical_profile" in data:
            profile.medical_profile = data["medical_profile"]
            profile_dirty_fields.append("medical_profile")
            if "medical_history" not in data:
                profile.medical_history = data["medical_profile"]
                profile_dirty_fields.append("medical_history")
        if "medical_history" in data and "medical_profile" not in data:
            profile.medical_history = data["medical_history"]
            profile.medical_profile = data["medical_history"]
            profile_dirty_fields.extend(["medical_history", "medical_profile"])
        if profile_dirty_fields:
            profile.save(update_fields=profile_dirty_fields + ["updated_at"])

        return Response(UserProfileSerializer(profile).data, status=status.HTTP_200_OK)


class ChatSessionListCreateView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [JSONParser]

    def get(self, request):
        q = request.query_params.get("q", "").strip()
        date_from = parse_date(request.query_params.get("date_from", "") or "")
        date_to = parse_date(request.query_params.get("date_to", "") or "")

        sessions = ChatSession.objects.filter(user=request.user)
        if q:
            sessions = sessions.filter(
                Q(title__icontains=q)
                | Q(messages__content__icontains=q)
                | Q(cases__symptom_text__icontains=q)
            )
        if date_from:
            sessions = sessions.filter(created_at__date__gte=date_from)
        if date_to:
            sessions = sessions.filter(created_at__date__lte=date_to)

        sessions = sessions.distinct().order_by("-updated_at")
        return Response(ChatSessionSerializer(sessions, many=True).data, status=status.HTTP_200_OK)

    def post(self, request):
        title = str(request.data.get("title", "")).strip() or "Health Consultation"
        session = ChatSession.objects.create(user=request.user, title=title)
        return Response(ChatSessionSerializer(session).data, status=status.HTTP_201_CREATED)


class ChatSessionMessagesView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, session_id: int):
        session = _session_for_user(request.user, session_id=session_id)
        return Response(
            {
                "session": ChatSessionSerializer(session).data,
                "messages": ChatMessageSerializer(session.messages.all(), many=True).data,
            },
            status=status.HTTP_200_OK,
        )


class ChatAnalyzeView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]
    throttle_classes = [AnalyzeRateThrottle]

    @transaction.atomic
    def post(self, request, session_id: int):
        session = _session_for_user(request.user, session_id=session_id)
        serializer = AnalyzeCaseSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data
        preferred_language = (
            payload.get("language_override")
            or getattr(request, "preferred_language", "")
            or str(_profile_for_user(request.user).preferred_language or "").strip().lower()
        )
        response_language = detect_language(payload["symptom_text"], preferred=preferred_language)

        # REQ-9: Redact PII from user message content before persisting to DB
        _redacted = redact_phi_text(payload["symptom_text"])
        stored_content = str(_redacted.get("redacted_text") or payload["symptom_text"])

        user_message = ChatMessage.objects.create(
            session=session,
            role=ChatMessage.Role.USER,
            content=stored_content,
            metadata={
                "symptom_tags": payload.get("symptom_tags", []),
                "language": response_language,
                "pii_redacted": bool(_redacted.get("entities")),
            },
        )

        profile = _profile_for_user(request.user)
        case_metadata = _profile_medical_payload(profile)
        case_metadata.update(payload.get("metadata", {}))
        case_metadata["language"] = response_language
        case_metadata["force_search"] = bool(payload.get("force_search", False))
        case_metadata["search_consent_given"] = bool(payload.get("search_consent_given", False))
        case_metadata["model_profile"] = payload.get("model_profile", "Clinical Balanced")
        if payload.get("mock_search_results"):
            case_metadata["mock_search_results"] = payload["mock_search_results"]

        case = CaseSubmission.objects.create(
            user=request.user,
            chat_session=session,
            symptom_text=payload["symptom_text"],
            symptom_tags=payload.get("symptom_tags", []),
            uploaded_image=payload.get("image"),
            consent_given=payload["consent_given"],
            location_lat=payload.get("location_lat"),
            location_lng=payload.get("location_lng"),
            facility_type_requested=payload.get("facility_type", ""),
            specialization_requested=payload.get("specialization", ""),
            search_radius_km=payload.get("search_radius_km", 5),
            metadata=case_metadata,
            status="processing",
        )

        try:
            result = run_case_analysis(case)
        except Exception:
            logger.exception("Chat analysis pipeline failed for case_id=%s", case.id)
            case.status = "failed"
            case.save(update_fields=["status"])
            return Response(
                {
                    "error": (
                        "መረጃውን ማብራራት አልቻልንም። እባክዎ እንደገና ይሞክሩ፤ ምልክቶቹ ከባድ ከሆኑ ወይም እየባሱ ከሄዱ የሙያ ሕክምና እርዳታ ይፈልጉ።"
                        if response_language == "am"
                        else "We could not complete this analysis. Please retry, and seek professional care if symptoms are severe or worsening."
                    )
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        assistant_message = ChatMessage.objects.create(
            session=session,
            role=ChatMessage.Role.ASSISTANT,
            content=_assistant_summary(result, symptom_text=payload["symptom_text"], language=response_language),
            metadata={
                "case_id": case.id,
                "result": result,
                "language": response_language,
                "created_at": timezone.now().isoformat(),
            },
        )

        # REQ-8: Send emergency email alert for High-risk cases
        if result.get("risk_level") == "High" or result.get("needs_urgent_care"):
            try:
                django_rq.enqueue(
                    send_emergency_alert_email,
                    request.user,
                    str(result.get("risk_level", "High")),
                    list(result.get("red_flags", [])),
                    case.id,
                )
                # Also alert emergency contact if configured
                profile = _profile_for_user(request.user)
                if profile.emergency_contact_phone and profile.emergency_contact_name:
                    # Use email field from profile if available (stored in medical_profile)
                    ec_email = (profile.medical_profile or {}).get("emergency_contact_email", "")
                    if ec_email:
                        patient_name = (
                            f"{request.user.first_name} {request.user.last_name}".strip()
                            or request.user.username
                        )
                        django_rq.enqueue(
                            send_emergency_contact_alert,
                            profile.emergency_contact_name,
                            ec_email,
                            patient_name,
                            str(result.get("risk_level", "High")),
                            case.id,
                        )
            except Exception:
                logger.warning("Emergency alert email failed for case_id=%s", case.id)
        session.updated_at = timezone.now()
        if session.title == "Health Consultation":
            session.title = (payload["symptom_text"][:64] + "...") if len(payload["symptom_text"]) > 64 else payload["symptom_text"]
            session.save(update_fields=["title", "updated_at"])
        else:
            session.save(update_fields=["updated_at"])

        return Response(
                {
                    "session": ChatSessionSerializer(session).data,
                    "user_message": ChatMessageSerializer(user_message).data,
                "assistant_message": ChatMessageSerializer(assistant_message).data,
                "analysis": result,
            },
            status=status.HTTP_200_OK,
        )


class AnalyzeCaseView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    throttle_classes = [AnalyzeAnonRateThrottle]

    @transaction.atomic
    def post(self, request):
        serializer = AnalyzeCaseSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data
        user = request.user if request.user.is_authenticated else None
        preferred_language = payload.get("language_override") or getattr(request, "preferred_language", "")
        if user:
            preferred_language = preferred_language or str(_profile_for_user(user).preferred_language or "").strip().lower()
        response_language = detect_language(payload["symptom_text"], preferred=preferred_language)
        profile_metadata = {}
        if user:
            profile = _profile_for_user(user)
            profile_metadata = _profile_medical_payload(profile)
        profile_metadata.update(payload.get("metadata", {}))
        profile_metadata["language"] = response_language
        profile_metadata["force_search"] = bool(payload.get("force_search", False))
        profile_metadata["search_consent_given"] = bool(payload.get("search_consent_given", False))
        profile_metadata["model_profile"] = payload.get("model_profile", "Clinical Balanced")
        if payload.get("mock_search_results"):
            profile_metadata["mock_search_results"] = payload["mock_search_results"]

        case = CaseSubmission.objects.create(
            user=user,
            symptom_text=payload["symptom_text"],
            symptom_tags=payload.get("symptom_tags", []),
            uploaded_image=payload.get("image"),
            consent_given=payload["consent_given"],
            location_lat=payload.get("location_lat"),
            location_lng=payload.get("location_lng"),
            facility_type_requested=payload.get("facility_type", ""),
            specialization_requested=payload.get("specialization", ""),
            search_radius_km=payload.get("search_radius_km", 5),
            metadata=profile_metadata,
            status="processing",
        )

        if payload.get("async_mode", False):
            submit_async_case_analysis(case.id)
            return Response(
                {
                    "case_id": case.id,
                    "status": "queued",
                    "poll_url": f"/api/v1/analyze/{case.id}/?token={case.status_token}",
                    "status_token": str(case.status_token),
                    "response_language": response_language,
                },
                status=status.HTTP_202_ACCEPTED,
            )

        try:
            result = run_case_analysis(case)
            return Response(result, status=status.HTTP_200_OK)
        except Exception:
            logger.exception("Analysis pipeline failed for case_id=%s", case.id)
            case.status = "failed"
            case.save(update_fields=["status"])
            return Response(
                {
                    "error": (
                        "መረጃውን ማብራራት አልቻልንም። እባክዎ እንደገና ይሞክሩ፤ ምልክቶቹ ከባድ ከሆኑ ወይም እየባሱ ከሄዱ የሙያ ሕክምና እርዳታ ይፈልጉ።"
                        if response_language == "am"
                        else "We could not complete this analysis. Please retry, and seek professional care if symptoms are severe or worsening."
                    )
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class AnalysisStatusView(APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def get(self, request, case_id: int):
        case = get_object_or_404(CaseSubmission, id=case_id)
        if case.user_id:
            if not request.user.is_authenticated or (request.user.id != case.user_id and not request.user.is_staff):
                return Response({"error": "Case not found."}, status=status.HTTP_404_NOT_FOUND)
        else:
            status_token = str(request.query_params.get("token", "")).strip()
            if not status_token or not constant_time_compare(status_token, str(case.status_token)):
                return Response({"error": "Case not found."}, status=status.HTTP_404_NOT_FOUND)

        task_status = async_case_status(case_id)
        payload = {
            "case_id": case.id,
            "status": case.status,
            "task_status": task_status,
        }
        if case.status == "completed":
            payload["result"] = _serialize_case_result(case)
        return Response(payload, status=status.HTTP_200_OK)


class ChatHistoryView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        q = request.query_params.get("q", "").strip()
        date_from = parse_date(request.query_params.get("date_from", "") or "")
        date_to = parse_date(request.query_params.get("date_to", "") or "")

        sessions = ChatSession.objects.filter(user=request.user)
        if q:
            sessions = sessions.filter(
                Q(title__icontains=q) | Q(messages__content__icontains=q) | Q(cases__symptom_text__icontains=q)
            )
        if date_from:
            sessions = sessions.filter(updated_at__date__gte=date_from)
        if date_to:
            sessions = sessions.filter(updated_at__date__lte=date_to)
        sessions = sessions.distinct().order_by("-updated_at")

        payload = []
        for session in sessions:
            payload.append(
                {
                    "session": ChatSessionSerializer(session).data,
                    "messages": ChatMessageSerializer(session.messages.all(), many=True).data,
                    "cases": [_serialize_case_result(case) for case in session.cases.all()],
                }
            )
        return Response(payload, status=status.HTTP_200_OK)


class LocationNearbyView(APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def get(self, request):
        serializer = FacilitySearchSerializer(data=request.query_params)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        facilities = lookup_nearby_facilities(
            location_lat=data["location_lat"],
            location_lng=data["location_lng"],
            facility_type=data.get("facility_type", "hospital"),
            specialization=data.get("specialization", ""),
            radius_km=data.get("radius_km", 5),
            limit=10,
        )
        return Response({"facilities": facilities}, status=status.HTTP_200_OK)


class LocationDirectionsView(APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def get(self, request):
        origin_lat = request.query_params.get("origin_lat")
        origin_lng = request.query_params.get("origin_lng")
        destination_lat = request.query_params.get("destination_lat")
        destination_lng = request.query_params.get("destination_lng")
        place_id = request.query_params.get("place_id", "").strip()

        if place_id:
            url = f"https://www.google.com/maps/dir/?api=1&destination_place_id={place_id}"
            return Response({"maps_url": url}, status=status.HTTP_200_OK)

        if not all([origin_lat, origin_lng, destination_lat, destination_lng]):
            return Response(
                {"error": "Provide place_id or origin/destination coordinates."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        url = (
            "https://www.google.com/maps/dir/?api=1"
            f"&origin={origin_lat},{origin_lng}&destination={destination_lat},{destination_lng}"
        )
        return Response({"maps_url": url}, status=status.HTTP_200_OK)


class EmergencyContactsView(APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def get(self, request):
        country_code = str(request.query_params.get("country_code", "")).strip().upper()
        return Response({"contacts": emergency_contacts(country_code=country_code)}, status=status.HTTP_200_OK)


class AdminUsersView(APIView):
    permission_classes = [permissions.IsAdminUser]

    def get(self, request):
        q = request.query_params.get("q", "").strip()
        page = max(int(request.query_params.get("page", 1)), 1)
        page_size = min(int(request.query_params.get("page_size", 50)), 200)

        users = User.objects.all().select_related("profile").order_by("-date_joined")
        if q:
            users = users.filter(
                Q(username__icontains=q) | Q(email__icontains=q) |
                Q(first_name__icontains=q) | Q(last_name__icontains=q)
            )
        total = users.count()
        page_users = users[(page - 1) * page_size: page * page_size]
        return Response(
            {
                "count": total,
                "page": page,
                "page_size": page_size,
                "results": [UserSerializer(user).data for user in page_users],
            },
            status=status.HTTP_200_OK,
        )


class AdminUserDetailView(APIView):
    permission_classes = [permissions.IsAdminUser]
    parser_classes = [JSONParser]

    def patch(self, request, user_id: int):
        user = get_object_or_404(User, id=user_id)
        serializer = AdminUserUpdateSerializer(data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        dirty_fields = []
        for field in ("first_name", "last_name", "email", "is_active", "is_staff"):
            if field in data:
                setattr(user, field, data[field])
                dirty_fields.append(field)
        if dirty_fields:
            user.save(update_fields=dirty_fields)
            AuditLog.objects.create(
                actor=request.user,
                action="admin_update_user",
                target_type="user",
                target_id=str(user.id),
                metadata={"fields": dirty_fields},
            )
        return Response(UserSerializer(user).data, status=status.HTTP_200_OK)


class AdminFacilitiesView(APIView):
    permission_classes = [permissions.IsAdminUser]
    parser_classes = [JSONParser]

    def get(self, request):
        facilities = HealthcareFacility.objects.all().order_by("name")
        return Response(HealthcareFacilitySerializer(facilities, many=True).data, status=status.HTTP_200_OK)

    def post(self, request):
        serializer = HealthcareFacilitySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        facility = serializer.save(source="manual")
        AuditLog.objects.create(
            actor=request.user,
            action="admin_create_facility",
            target_type="facility",
            target_id=str(facility.id),
        )
        return Response(HealthcareFacilitySerializer(facility).data, status=status.HTTP_201_CREATED)


class AdminFacilityDetailView(APIView):
    permission_classes = [permissions.IsAdminUser]
    parser_classes = [JSONParser]

    def patch(self, request, facility_id: int):
        facility = get_object_or_404(HealthcareFacility, id=facility_id)
        serializer = HealthcareFacilitySerializer(instance=facility, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        facility = serializer.save()
        AuditLog.objects.create(
            actor=request.user,
            action="admin_update_facility",
            target_type="facility",
            target_id=str(facility.id),
        )
        return Response(HealthcareFacilitySerializer(facility).data, status=status.HTTP_200_OK)

    def delete(self, request, facility_id: int):
        facility = get_object_or_404(HealthcareFacility, id=facility_id)
        facility.delete()
        AuditLog.objects.create(
            actor=request.user,
            action="admin_delete_facility",
            target_type="facility",
            target_id=str(facility_id),
        )
        return Response(status=status.HTTP_204_NO_CONTENT)


class AdminAnalyticsView(APIView):
    permission_classes = [permissions.IsAdminUser]

    def get(self, request):
        today = timezone.now().date()
        risk_breakdown = (
            CaseSubmission.objects.filter(risk__isnull=False)
            .values("risk__risk_level")
            .annotate(count=Count("id"))
            .order_by("risk__risk_level")
        )
        return Response(
            {
                "users_total": User.objects.count(),
                "users_active": User.objects.filter(is_active=True).count(),
                "chat_sessions_total": ChatSession.objects.count(),
                "chat_messages_total": ChatMessage.objects.count(),
                "cases_total": CaseSubmission.objects.count(),
                "cases_today": CaseSubmission.objects.filter(created_at__date=today).count(),
                "cases_completed": CaseSubmission.objects.filter(status="completed").count(),
                "facilities_total": HealthcareFacility.objects.count(),
                "risk_breakdown": list(risk_breakdown),
            },
            status=status.HTTP_200_OK,
        )


class AdminConfigView(APIView):
    permission_classes = [permissions.IsAdminUser]
    parser_classes = [JSONParser]

    def post(self, request):
        action = str(request.data.get("action", "")).strip()
        if action == "retrain_text_model":
            AuditLog.objects.create(
                actor=request.user,
                action="admin_retrain_requested",
                target_type="ml_model",
                target_id="text",
            )
            
            import django_rq
            import subprocess
            from django.conf import settings
            
            def run_retrain_script():
                subprocess.run(["python", str(settings.BASE_DIR / "scripts" / "preprocess_and_train.py")])
                
            django_rq.enqueue(run_retrain_script)
            
            return Response(
                {
                    "status": "queued",
                    "message": "Model retraining task pushed to the asynchronous worker queue successfully. Processing in background.",
                },
                status=status.HTTP_202_ACCEPTED,
            )
        return Response(
            {"error": "Unsupported action. Example: retrain_text_model"},
            status=status.HTTP_400_BAD_REQUEST,
        )


class ExportProfileView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        profile_data = UserProfileSerializer(_profile_for_user(request.user)).data
        cases = [_serialize_case_result(case) for case in request.user.cases.all()[:200]]
        return Response(
            {
                "exported_at": timezone.now().isoformat(),
                "user": UserSerializer(request.user).data,
                "profile": profile_data,
                "cases": cases,
            },
            status=status.HTTP_200_OK,
        )


class ExportChatHistoryView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        output_format = (request.query_params.get("format", "json") or "json").strip().lower()
        session_id = request.query_params.get("session_id")
        sessions = ChatSession.objects.filter(user=request.user).order_by("-updated_at")
        if session_id:
            sessions = sessions.filter(id=session_id)

        if output_format == "csv":
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(["session_id", "session_title", "role", "content", "created_at"])
            for session in sessions:
                for message in session.messages.all():
                    writer.writerow([session.id, session.title, message.role, message.content, message.created_at.isoformat()])
            response = HttpResponse(csv_buffer.getvalue(), content_type="text/csv")
            response["Content-Disposition"] = 'attachment; filename="chat_history.csv"'
            return response

        payload = []
        for session in sessions:
            payload.append(
                {
                    "session": ChatSessionSerializer(session).data,
                    "messages": ChatMessageSerializer(session.messages.all(), many=True).data,
                    "cases": [_serialize_case_result(case) for case in session.cases.all()],
                }
            )
        return Response(
            {
                "exported_at": timezone.now().isoformat(),
                "sessions": payload,
            },
            status=status.HTTP_200_OK,
        )


# ── REQ-7: SSE real-time job status stream ──────────────────────────────────

class AnalysisStatusSSEView(APIView):
    """
    Server-Sent Events endpoint for real-time async job status.
    GET /api/v1/analyze/<case_id>/stream/?token=<uuid>
    Streams status events until the job reaches a terminal state.
    """
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def get(self, request, case_id: int):
        case = get_object_or_404(CaseSubmission, id=case_id)
        if case.user_id:
            if not request.user.is_authenticated or (
                request.user.id != case.user_id and not request.user.is_staff
            ):
                return Response({"error": "Case not found."}, status=status.HTTP_404_NOT_FOUND)
        else:
            status_token = str(request.query_params.get("token", "")).strip()
            if not status_token or not constant_time_compare(status_token, str(case.status_token)):
                return Response({"error": "Case not found."}, status=status.HTTP_404_NOT_FOUND)

        def _event_stream():
            terminal = {"completed", "failed", "unknown"}
            poll_interval = float(getattr(settings, "SSE_POLL_INTERVAL_SECONDS", 1.5))
            max_wait = float(getattr(settings, "SSE_MAX_WAIT_SECONDS", 620))
            elapsed = 0.0
            last_status = None
            while elapsed < max_wait:
                current_case = CaseSubmission.objects.filter(id=case_id).only("status", "async_job_id").first()
                if current_case is None:
                    yield f"data: {json.dumps({'status': 'unknown', 'case_id': case_id})}\n\n"
                    break
                task_status = async_case_status(case_id)
                if task_status != last_status:
                    payload: dict = {"case_id": case_id, "status": task_status}
                    if task_status == "completed":
                        payload["result"] = _serialize_case_result(current_case)
                    yield f"data: {json.dumps(payload)}\n\n"
                    last_status = task_status
                if task_status in terminal:
                    break
                time.sleep(poll_interval)
                elapsed += poll_interval
            else:
                yield f"data: {json.dumps({'case_id': case_id, 'status': 'timeout'})}\n\n"

        response = StreamingHttpResponse(_event_stream(), content_type="text/event-stream")
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"
        return response


# ── REQ-5: Admin user management (list + detail already exist) ───────────────
# AdminAuditLogView — expose audit log to admin UI

class AdminAuditLogView(APIView):
    """GET /api/v1/admin/audit-log/ — paginated audit log for admin dashboard."""
    permission_classes = [permissions.IsAdminUser]

    def get(self, request):
        from .models import AuditLog as _AuditLog
        qs = _AuditLog.objects.select_related("actor").order_by("-created_at")
        actor_id = request.query_params.get("actor_id")
        action = request.query_params.get("action", "").strip()
        if actor_id:
            qs = qs.filter(actor_id=actor_id)
        if action:
            qs = qs.filter(action__icontains=action)
        page_size = min(int(request.query_params.get("page_size", 50)), 200)
        page = max(int(request.query_params.get("page", 1)), 1)
        total = qs.count()
        entries = qs[(page - 1) * page_size: page * page_size]
        results = [
            {
                "id": entry.id,
                "actor": entry.actor.username if entry.actor else None,
                "action": entry.action,
                "target_type": entry.target_type,
                "target_id": entry.target_id,
                "metadata": entry.metadata,
                "created_at": entry.created_at.isoformat(),
            }
            for entry in entries
        ]
        return Response({"total": total, "page": page, "page_size": page_size, "results": results})


# ── REQ-1: Email verification ────────────────────────────────────────────────

class VerifyEmailView(APIView):
    """
    GET /api/v1/auth/verify-email/?token=<uuid>
    Marks the user's email as verified.
    """
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def get(self, request):
        from .models import EmailVerificationToken
        token_str = str(request.query_params.get("token", "")).strip()
        if not token_str:
            return Response({"error": "Token is required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            ev = EmailVerificationToken.objects.select_related("user").get(token=token_str)
        except EmailVerificationToken.DoesNotExist:
            return Response({"error": "Invalid or expired token."}, status=status.HTTP_400_BAD_REQUEST)
        if ev.is_verified:
            return Response({"status": "already_verified"}, status=status.HTTP_200_OK)
        ev.verified_at = timezone.now()
        ev.save(update_fields=["verified_at"])
        AuditLog.objects.create(
            actor=ev.user,
            action="email_verified",
            target_type="user",
            target_id=str(ev.user_id),
        )
        return Response({"status": "verified", "email": ev.user.email}, status=status.HTTP_200_OK)


class ResendVerificationEmailView(APIView):
    """
    POST /api/v1/auth/resend-verification/
    Resends the verification email for the authenticated user.
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        from .models import EmailVerificationToken
        user = request.user
        ev_token, _ = EmailVerificationToken.objects.get_or_create(user=user)
        if ev_token.is_verified:
            return Response({"status": "already_verified"}, status=status.HTTP_200_OK)
        sent = send_verification_email(user, str(ev_token.token))
        return Response({"status": "sent" if sent else "queued"}, status=status.HTTP_200_OK)


# ── REQ-1: Password Reset ────────────────────────────────────────────────────

class PasswordResetRequestView(APIView):
    """
    POST /api/v1/auth/password-reset/
    Body: { "email": "user@example.com" }
    Sends a password reset link. Always returns 200 to prevent email enumeration.
    """
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    parser_classes = [JSONParser]
    throttle_classes = []  # Axes + auth throttle applied at settings level

    def post(self, request):
        from .models import PasswordResetToken
        from .services.email_service import send_password_reset_email
        email = str(request.data.get("email", "")).strip().lower()
        if email:
            user = User.objects.filter(email__iexact=email, is_active=True).first()
            if user:
                # Invalidate any existing unused tokens
                PasswordResetToken.objects.filter(user=user, used_at__isnull=True).delete()
                token = PasswordResetToken.objects.create(user=user)
                send_password_reset_email(user, str(token.token))
                AuditLog.objects.create(
                    actor=None,
                    action="password_reset_requested",
                    target_type="user",
                    target_id=str(user.id),
                )
        # Always return 200 — never reveal whether email exists
        return Response(
            {"status": "ok", "message": "If that email is registered, a reset link has been sent."},
            status=status.HTTP_200_OK,
        )


class PasswordResetConfirmView(APIView):
    """
    POST /api/v1/auth/password-reset/confirm/
    Body: { "token": "<uuid>", "new_password": "NewPass123" }
    """
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    parser_classes = [JSONParser]

    def post(self, request):
        from .models import PasswordResetToken
        token_str = str(request.data.get("token", "")).strip()
        new_password = str(request.data.get("new_password", "")).strip()

        if not token_str or not new_password:
            return Response(
                {"error": "token and new_password are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Password strength validation
        import re as _re
        if len(new_password) < 8:
            return Response({"error": "Password must be at least 8 characters."}, status=status.HTTP_400_BAD_REQUEST)
        if not _re.search(r"[A-Z]", new_password):
            return Response({"error": "Password must include an uppercase letter."}, status=status.HTTP_400_BAD_REQUEST)
        if not _re.search(r"[a-z]", new_password):
            return Response({"error": "Password must include a lowercase letter."}, status=status.HTTP_400_BAD_REQUEST)
        if not _re.search(r"[0-9]", new_password):
            return Response({"error": "Password must include a number."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            reset_token = PasswordResetToken.objects.select_related("user").get(token=token_str)
        except PasswordResetToken.DoesNotExist:
            return Response({"error": "Invalid or expired reset token."}, status=status.HTTP_400_BAD_REQUEST)

        if reset_token.is_used:
            return Response({"error": "This reset link has already been used."}, status=status.HTTP_400_BAD_REQUEST)
        if reset_token.is_expired:
            return Response({"error": "This reset link has expired. Please request a new one."}, status=status.HTTP_400_BAD_REQUEST)

        user = reset_token.user
        user.set_password(new_password)
        user.save(update_fields=["password"])
        reset_token.used_at = timezone.now()
        reset_token.save(update_fields=["used_at"])

        # Blacklist all existing refresh tokens for this user
        try:
            from rest_framework_simplejwt.token_blacklist.models import OutstandingToken
            for outstanding in OutstandingToken.objects.filter(user=user):
                try:
                    outstanding.blacklistedtoken  # already blacklisted
                except Exception:
                    from rest_framework_simplejwt.token_blacklist.models import BlacklistedToken
                    BlacklistedToken.objects.get_or_create(token=outstanding)
        except Exception:
            pass

        AuditLog.objects.create(
            actor=user,
            action="password_reset_completed",
            target_type="user",
            target_id=str(user.id),
        )
        return Response({"status": "ok", "message": "Password updated successfully."}, status=status.HTTP_200_OK)


# ── REQ-5: Admin dialogue template editor ───────────────────────────────────

class AdminDialogueTemplatesView(APIView):
    """
    GET  /api/v1/admin/dialogue-templates/  — list all response templates
    POST /api/v1/admin/dialogue-templates/  — update a template key
    Body: { "intent": "<intent_key>", "templates": ["response1", "response2"] }
    """
    permission_classes = [permissions.IsAdminUser]
    parser_classes = [JSONParser]

    def _load_templates(self) -> dict:
        import json as _json
        path = getattr(settings, "DIALOGUE_RESPONSE_TEMPLATES_PATH", "")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return _json.load(f)
        except Exception:
            return {}

    def _save_templates(self, data: dict) -> bool:
        import json as _json
        path = getattr(settings, "DIALOGUE_RESPONSE_TEMPLATES_PATH", "")
        try:
            with open(path, "w", encoding="utf-8") as f:
                _json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    def get(self, request):
        templates = self._load_templates()
        return Response({"templates": templates}, status=status.HTTP_200_OK)

    def post(self, request):
        intent = str(request.data.get("intent", "")).strip()
        new_templates = request.data.get("templates")
        if not intent:
            return Response({"error": "intent is required."}, status=status.HTTP_400_BAD_REQUEST)
        if not isinstance(new_templates, list) or not all(isinstance(t, str) for t in new_templates):
            return Response({"error": "templates must be a list of strings."}, status=status.HTTP_400_BAD_REQUEST)

        data = self._load_templates()
        data[intent] = new_templates
        if not self._save_templates(data):
            return Response({"error": "Unable to save templates file."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        AuditLog.objects.create(
            actor=request.user,
            action="admin_update_dialogue_template",
            target_type="dialogue_template",
            target_id=intent,
            metadata={"template_count": len(new_templates)},
        )
        return Response({"status": "updated", "intent": intent, "templates": new_templates}, status=status.HTTP_200_OK)


# ── REQ-5: Admin model performance metrics ──────────────────────────────────

class AdminModelMetricsView(APIView):
    """
    GET /api/v1/admin/model-metrics/
    Returns text and image model evaluation metrics from stored JSON files.
    """
    permission_classes = [permissions.IsAdminUser]

    def get(self, request):
        import json as _json
        from pathlib import Path as _Path

        base = _Path(settings.BASE_DIR) / "models"

        def _load(path: _Path) -> dict:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return _json.load(f)
            except Exception:
                return {}

        text_metrics = _load(base / "text_training_metrics.json")
        text_eval = _load(base / "evaluation" / "text_evaluation_summary.json")
        image_metrics = _load(base / "image_training_metrics.json")
        dialogue_metrics = _load(base / "dialogue_training_metrics.json")

        return Response(
            {
                "text_model": {
                    "training": text_metrics,
                    "evaluation": text_eval,
                },
                "image_model": {
                    "training": image_metrics,
                },
                "dialogue_model": {
                    "training": dialogue_metrics,
                },
            },
            status=status.HTTP_200_OK,
        )

class AdminRetrainModelView(APIView):
    permission_classes = [permissions.IsAdminUser]

    def post(self, request):
        import subprocess
        # Async trigger for the retraining script
        script_path = settings.BASE_DIR / "scripts" / "preprocess_and_train.py"
        try:
            subprocess.Popen(["python", str(script_path)])
            return Response({"status": "Retraining job queued and started in background."}, status=status.HTTP_202_ACCEPTED)
        except Exception as e:
            return Response({"error": f"Failed to start retraining: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
