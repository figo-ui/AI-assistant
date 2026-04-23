import uuid

from django.contrib.auth.models import User
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver


class UserProfile(models.Model):
    class Gender(models.TextChoices):
        MALE = "male", "Male"
        FEMALE = "female", "Female"
        OTHER = "other", "Other"
        PREFER_NOT_TO_SAY = "na", "Prefer not to say"

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    phone_number = models.CharField(max_length=32, blank=True)
    age = models.PositiveIntegerField(null=True, blank=True)
    gender = models.CharField(max_length=16, choices=Gender.choices, blank=True)
    address = models.CharField(max_length=255, blank=True)
    emergency_contact_name = models.CharField(max_length=120, blank=True)
    emergency_contact_phone = models.CharField(max_length=32, blank=True)
    profile_photo = models.ImageField(upload_to="profile_photos/", null=True, blank=True)
    medical_history = models.JSONField(default=dict, blank=True)
    medical_profile = models.JSONField(default=dict, blank=True)
    preferred_language = models.CharField(max_length=16, default="en")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"UserProfile(user_id={self.user_id})"


class ChatSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="chat_sessions")
    title = models.CharField(max_length=255, default="Health Consultation")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return f"ChatSession(id={self.id}, user_id={self.user_id})"


class ChatMessage(models.Model):
    class Role(models.TextChoices):
        USER = "user", "User"
        ASSISTANT = "assistant", "Assistant"
        SYSTEM = "system", "System"

    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=16, choices=Role.choices)
    content = models.TextField(blank=True)
    uploaded_image = models.ImageField(upload_to="chat_uploads/", null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"ChatMessage(id={self.id}, role={self.role}, session_id={self.session_id})"


class HealthcareFacility(models.Model):
    class FacilityType(models.TextChoices):
        HOSPITAL = "hospital", "Hospital"
        CLINIC = "clinic", "Clinic"
        PHARMACY = "pharmacy", "Pharmacy"
        EMERGENCY = "emergency", "Emergency"

    name = models.CharField(max_length=255)
    facility_type = models.CharField(
        max_length=24,
        choices=FacilityType.choices,
        default=FacilityType.HOSPITAL,
    )
    specialization = models.CharField(max_length=120, blank=True)
    address = models.CharField(max_length=512, blank=True)
    phone_number = models.CharField(max_length=32, blank=True)
    latitude = models.FloatField()
    longitude = models.FloatField()
    is_emergency = models.BooleanField(default=False)
    source = models.CharField(max_length=32, default="manual")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return f"HealthcareFacility(id={self.id}, name={self.name})"


class CaseSubmission(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name="cases")
    chat_session = models.ForeignKey(
        ChatSession,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="cases",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    symptom_text = models.TextField()
    symptom_tags = models.JSONField(default=list, blank=True)
    uploaded_image = models.ImageField(upload_to="case_uploads/", null=True, blank=True)
    consent_given = models.BooleanField(default=False)
    location_lat = models.FloatField(null=True, blank=True)
    location_lng = models.FloatField(null=True, blank=True)
    facility_type_requested = models.CharField(max_length=24, blank=True)
    specialization_requested = models.CharField(max_length=120, blank=True)
    search_radius_km = models.PositiveIntegerField(default=5)
    status = models.CharField(max_length=32, default="received")
    status_token = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, db_index=True)
    async_job_id = models.CharField(max_length=64, blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"CaseSubmission(id={self.id}, status={self.status})"


class InferenceRecord(models.Model):
    case = models.OneToOneField(CaseSubmission, on_delete=models.CASCADE, related_name="inference")
    text_predictions = models.JSONField(default=list, blank=True)
    image_predictions = models.JSONField(default=list, blank=True)
    fused_predictions = models.JSONField(default=list, blank=True)
    text_confidence = models.FloatField(null=True, blank=True)
    image_confidence = models.FloatField(null=True, blank=True)
    fusion_confidence = models.FloatField(null=True, blank=True)
    confidence_band = models.CharField(max_length=16, default="low")
    text_model_version = models.CharField(max_length=64, default="text-v1")
    image_model_version = models.CharField(max_length=64, default="image-v1")
    fusion_version = models.CharField(max_length=64, default="fusion-v1")
    uncertainty = models.FloatField(default=1.0)
    disagreement = models.FloatField(default=0.0)
    latency_ms = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"InferenceRecord(case_id={self.case_id}, confidence={self.fusion_confidence})"


class RiskAssessment(models.Model):
    class RiskLevel(models.TextChoices):
        LOW = "Low", "Low"
        MEDIUM = "Medium", "Medium"
        HIGH = "High", "High"

    case = models.OneToOneField(CaseSubmission, on_delete=models.CASCADE, related_name="risk")
    risk_score = models.FloatField()
    risk_level = models.CharField(max_length=16, choices=RiskLevel.choices)
    severity_component = models.FloatField(default=0.0)
    redflag_component = models.FloatField(default=0.0)
    vulnerability_component = models.FloatField(default=0.0)
    uncertainty_component = models.FloatField(default=0.0)
    disagreement_component = models.FloatField(default=0.0)
    recommendation_text = models.TextField()
    disclaimer_text = models.TextField()
    needs_urgent_care = models.BooleanField(default=False)
    red_flags = models.JSONField(default=list, blank=True)
    risk_factors = models.JSONField(default=list, blank=True)
    prevention_advice = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"RiskAssessment(case_id={self.case_id}, risk={self.risk_level})"


class FacilityResult(models.Model):
    case = models.ForeignKey(CaseSubmission, on_delete=models.CASCADE, related_name="facilities")
    provider_name = models.CharField(max_length=255)
    address = models.CharField(max_length=512, blank=True)
    place_id = models.CharField(max_length=128, blank=True)
    distance_km = models.FloatField(null=True, blank=True)
    rating = models.FloatField(null=True, blank=True)
    maps_url = models.URLField(blank=True)
    source = models.CharField(max_length=32, default="google_places")
    phone_number = models.CharField(max_length=32, blank=True)
    facility_type = models.CharField(max_length=24, blank=True)
    specialization = models.CharField(max_length=120, blank=True)
    is_emergency = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"FacilityResult(case_id={self.case_id}, provider={self.provider_name})"


class AuditLog(models.Model):
    actor = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    action = models.CharField(max_length=120)
    target_type = models.CharField(max_length=64, blank=True)
    target_id = models.CharField(max_length=64, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"AuditLog(id={self.id}, action={self.action})"


class EmailVerificationToken(models.Model):
    """One-time token for email address verification."""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="email_verification")
    token = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    verified_at = models.DateTimeField(null=True, blank=True)

    @property
    def is_verified(self) -> bool:
        return self.verified_at is not None

    def __str__(self) -> str:
        return f"EmailVerificationToken(user_id={self.user_id}, verified={self.is_verified})"


class PasswordResetToken(models.Model):
    """Single-use token for password reset (expires after 1 hour)."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="password_reset_tokens")
    token = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    used_at = models.DateTimeField(null=True, blank=True)

    @property
    def is_expired(self) -> bool:
        from django.utils import timezone
        from datetime import timedelta
        return (timezone.now() - self.created_at) > timedelta(hours=1)

    @property
    def is_used(self) -> bool:
        return self.used_at is not None

    def __str__(self) -> str:
        return f"PasswordResetToken(user_id={self.user_id}, used={self.is_used})"


@receiver(post_save, sender=User)
def create_profile_for_user(sender, instance: User, created: bool, **kwargs):
    if created:
        UserProfile.objects.get_or_create(user=instance)
