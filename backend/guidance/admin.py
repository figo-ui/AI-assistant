from django.contrib import admin

from .models import (
    AuditLog,
    CaseSubmission,
    ChatMessage,
    ChatSession,
    FacilityResult,
    HealthcareFacility,
    InferenceRecord,
    RiskAssessment,
    UserProfile,
)


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "phone_number", "age", "preferred_language", "updated_at")
    search_fields = ("user__username", "user__email", "phone_number")


@admin.register(CaseSubmission)
class CaseSubmissionAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "chat_session", "created_at", "status", "consent_given")
    search_fields = ("symptom_text",)
    list_filter = ("status", "consent_given", "created_at")


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "title", "created_at", "updated_at")
    search_fields = ("user__username", "title")


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ("id", "session", "role", "created_at")
    search_fields = ("content", "session__user__username")
    list_filter = ("role", "created_at")


@admin.register(InferenceRecord)
class InferenceRecordAdmin(admin.ModelAdmin):
    list_display = ("id", "case", "fusion_confidence", "confidence_band", "created_at")
    list_filter = ("confidence_band", "created_at")


@admin.register(RiskAssessment)
class RiskAssessmentAdmin(admin.ModelAdmin):
    list_display = ("id", "case", "risk_level", "risk_score", "needs_urgent_care", "created_at")
    list_filter = ("risk_level", "needs_urgent_care", "created_at")


@admin.register(FacilityResult)
class FacilityResultAdmin(admin.ModelAdmin):
    list_display = ("id", "case", "provider_name", "distance_km", "rating", "created_at")
    search_fields = ("provider_name", "address")


@admin.register(HealthcareFacility)
class HealthcareFacilityAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "facility_type", "specialization", "is_emergency", "updated_at")
    search_fields = ("name", "address", "specialization", "phone_number")
    list_filter = ("facility_type", "is_emergency")


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ("id", "actor", "action", "target_type", "target_id", "created_at")
    search_fields = ("action", "target_type", "target_id", "actor__username")
