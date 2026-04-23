"""
Custom DRF throttle classes for healthcare endpoints.
Stricter limits on auth and analysis endpoints.
"""
from rest_framework.throttling import AnonRateThrottle, UserRateThrottle


class AuthRateThrottle(AnonRateThrottle):
    """10 requests/minute on auth endpoints (register, password reset)."""
    scope = "auth"


class AnalyzeRateThrottle(UserRateThrottle):
    """20 analysis requests/minute per authenticated user."""
    scope = "analyze"


class AnalyzeAnonRateThrottle(AnonRateThrottle):
    """10 analysis requests/minute for anonymous users."""
    scope = "analyze"
