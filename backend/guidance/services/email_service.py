"""
REQ-1: Email verification and notification service.
Uses Django's built-in email backend — configure EMAIL_* settings in .env.
Falls back to console output when EMAIL_BACKEND is not configured.
"""
from __future__ import annotations

import logging
from typing import Optional

from django.conf import settings
from django.contrib.auth.models import User
from django.core.mail import send_mail
from django.utils import timezone

logger = logging.getLogger(__name__)

_FRONTEND_BASE = getattr(settings, "FRONTEND_BASE_URL", "http://127.0.0.1:5173")


def _send(subject: str, body: str, recipient: str) -> bool:
    """Send a plain-text email. Returns True on success."""
    from_email = getattr(settings, "DEFAULT_FROM_EMAIL", "noreply@healthcare-ai.local")
    try:
        send_mail(subject, body, from_email, [recipient], fail_silently=False)
        return True
    except Exception as exc:
        logger.warning("Email send failed to %s: %s", recipient, exc)
        return False


def send_verification_email(user: User, token_uuid: str) -> bool:
    """Send email address verification link."""
    verify_url = f"{_FRONTEND_BASE}/verify-email?token={token_uuid}"
    body = (
        f"Hello {user.first_name or user.username},\n\n"
        f"Please verify your email address by clicking the link below:\n\n"
        f"{verify_url}\n\n"
        f"This link is valid for 48 hours.\n\n"
        f"If you did not create this account, you can safely ignore this email.\n\n"
        f"— AI Healthcare Assistant"
    )
    return _send("Verify your email — AI Healthcare Assistant", body, user.email)


def send_password_reset_email(user: User, token_uuid: str) -> bool:
    """Send password reset link."""
    reset_url = f"{_FRONTEND_BASE}/reset-password?token={token_uuid}"
    body = (
        f"Hello {user.first_name or user.username},\n\n"
        f"A password reset was requested for your account.\n\n"
        f"Click the link below to reset your password:\n\n"
        f"{reset_url}\n\n"
        f"This link expires in 1 hour. If you did not request this, ignore this email.\n\n"
        f"— AI Healthcare Assistant"
    )
    return _send("Password reset — AI Healthcare Assistant", body, user.email)


def send_emergency_alert_email(user: User, risk_level: str, red_flags: list, case_id: int) -> bool:
    """REQ-8: Notify user by email when a High-risk case is detected."""
    if not user.email:
        return False
    flags_text = ", ".join(red_flags[:5]) if red_flags else "none"
    body = (
        f"Hello {user.first_name or user.username},\n\n"
        f"Your recent health assessment (Case #{case_id}) was flagged as HIGH RISK.\n\n"
        f"Risk level: {risk_level}\n"
        f"Red flags detected: {flags_text}\n\n"
        f"Please seek immediate in-person medical care or call your local emergency number.\n\n"
        f"This is an automated alert from AI Healthcare Assistant. "
        f"This is not a diagnosis — always consult a licensed medical professional.\n\n"
        f"— AI Healthcare Assistant"
    )
    return _send("URGENT: High-risk health alert — AI Healthcare Assistant", body, user.email)


def send_emergency_contact_alert(
    contact_name: str,
    contact_email: str,
    patient_name: str,
    risk_level: str,
    case_id: int,
) -> bool:
    """REQ-8: Notify emergency contact when patient has a High-risk case."""
    if not contact_email:
        return False
    body = (
        f"Hello {contact_name},\n\n"
        f"This is an automated alert from AI Healthcare Assistant.\n\n"
        f"{patient_name} recently completed a health assessment that was flagged as HIGH RISK "
        f"(Case #{case_id}, Risk Level: {risk_level}).\n\n"
        f"Please check on them and encourage them to seek immediate medical attention.\n\n"
        f"— AI Healthcare Assistant"
    )
    return _send(f"Health alert for {patient_name} — AI Healthcare Assistant", body, contact_email)
