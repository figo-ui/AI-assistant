import json
import re

from django.contrib.auth.models import User
from rest_framework import serializers

from .models import ChatMessage, ChatSession, HealthcareFacility, UserProfile


ALLOWED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class AnalyzeCaseSerializer(serializers.Serializer):
    symptom_text = serializers.CharField(max_length=5000, trim_whitespace=True)
    symptom_tags = serializers.ListField(
        child=serializers.CharField(max_length=60, trim_whitespace=True),
        required=False,
        allow_empty=True,
    )
    image = serializers.ImageField(required=False, allow_null=True)
    consent_given = serializers.BooleanField()
    model_profile = serializers.ChoiceField(
        choices=["Clinical Balanced", "Clinical Fast", "Clinical Thorough"],
        required=False,
    )
    location_lat = serializers.FloatField(required=False)
    location_lng = serializers.FloatField(required=False)
    metadata = serializers.JSONField(required=False)
    facility_type = serializers.ChoiceField(
        choices=[choice for choice, _ in HealthcareFacility.FacilityType.choices],
        required=False,
    )
    specialization = serializers.CharField(max_length=120, required=False, allow_blank=True)
    search_radius_km = serializers.IntegerField(required=False, min_value=1, max_value=50)
    language_override = serializers.ChoiceField(choices=["en", "am"], required=False)
    force_search = serializers.BooleanField(required=False)
    search_consent_given = serializers.BooleanField(required=False)
    async_mode = serializers.BooleanField(required=False)
    mock_search_results = serializers.ListField(child=serializers.DictField(), required=False, allow_empty=True)

    def validate_symptom_text(self, value: str) -> str:
        value = value.strip()
        if len(value) < 8:
            raise serializers.ValidationError("Please provide more symptom details.")
        return value

    def validate_metadata(self, value):
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise serializers.ValidationError("Metadata must be valid JSON.") from exc
            if not isinstance(parsed, dict):
                raise serializers.ValidationError("Metadata must be a JSON object.")
            return parsed
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise serializers.ValidationError("Metadata must be a JSON object.")
        return value

    def validate(self, attrs):
        if not attrs.get("consent_given", False):
            raise serializers.ValidationError(
                {"consent_given": "Consent is required before running AI guidance analysis."}
            )

        lat_present = "location_lat" in attrs
        lng_present = "location_lng" in attrs
        if lat_present ^ lng_present:
            raise serializers.ValidationError(
                "Both location_lat and location_lng are required when location is provided."
            )

        image = attrs.get("image")
        if image:
            if image.size > 5 * 1024 * 1024:
                raise serializers.ValidationError({"image": "Image size must be 5 MB or less."})
            lower_name = image.name.lower()
            if not lower_name.endswith(ALLOWED_IMAGE_EXTENSIONS):
                raise serializers.ValidationError({"image": "Only JPG and PNG images are supported."})

        tags = attrs.get("symptom_tags", [])
        attrs["symptom_tags"] = [tag.strip() for tag in tags if tag.strip()]
        attrs["metadata"] = attrs.get("metadata") or {}
        attrs["search_radius_km"] = int(attrs.get("search_radius_km", 5))
        attrs["force_search"] = bool(attrs.get("force_search", False))
        attrs["search_consent_given"] = bool(attrs.get("search_consent_given", False))
        attrs["async_mode"] = bool(attrs.get("async_mode", False))
        attrs["mock_search_results"] = attrs.get("mock_search_results") or []
        if "model_profile" not in attrs:
            attrs["model_profile"] = "Clinical Balanced"
        return attrs


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ("id", "username", "email", "first_name", "last_name", "is_active", "is_staff")


class UserProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    email_verified = serializers.SerializerMethodField()

    class Meta:
        model = UserProfile
        fields = (
            "user",
            "phone_number",
            "age",
            "gender",
            "address",
            "emergency_contact_name",
            "emergency_contact_phone",
            "medical_history",
            "medical_profile",
            "preferred_language",
            "email_verified",
            "created_at",
            "updated_at",
        )

    def get_email_verified(self, obj: UserProfile) -> bool:
        try:
            return obj.user.email_verification.is_verified
        except Exception:
            return False


class RegisterSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150, required=False, allow_blank=True)
    email = serializers.EmailField(required=True)
    first_name = serializers.CharField(max_length=150, required=False, allow_blank=True)
    last_name = serializers.CharField(max_length=150, required=False, allow_blank=True)
    password = serializers.CharField(write_only=True, min_length=8)
    phone_number = serializers.CharField(max_length=32, required=False, allow_blank=True)

    def validate_password(self, value: str) -> str:
        if len(value) < 8:
            raise serializers.ValidationError("Password must be at least 8 characters long.")
        if not re.search(r"[A-Z]", value):
            raise serializers.ValidationError("Password must include at least one uppercase letter.")
        if not re.search(r"[a-z]", value):
            raise serializers.ValidationError("Password must include at least one lowercase letter.")
        if not re.search(r"[0-9]", value):
            raise serializers.ValidationError("Password must include at least one number.")
        return value

    def validate_email(self, value: str) -> str:
        email = value.strip().lower()
        if User.objects.filter(email__iexact=email).exists():
            raise serializers.ValidationError("An account with this email already exists.")
        return email

    def validate_username(self, value: str) -> str:
        username = value.strip()
        if username and User.objects.filter(username__iexact=username).exists():
            raise serializers.ValidationError("This username is already taken.")
        return username

    def create(self, validated_data):
        email = validated_data["email"]
        username = validated_data.get("username", "").strip()
        if not username:
            base = email.split("@")[0][:24] or "user"
            username = base
            suffix = 1
            while User.objects.filter(username__iexact=username).exists():
                suffix += 1
                username = f"{base}{suffix}"

        user = User.objects.create_user(
            username=username,
            email=email,
            password=validated_data["password"],
            first_name=validated_data.get("first_name", "").strip(),
            last_name=validated_data.get("last_name", "").strip(),
        )
        profile = user.profile
        profile.phone_number = validated_data.get("phone_number", "").strip()
        profile.save(update_fields=["phone_number", "updated_at"])
        return user


class LoginSerializer(serializers.Serializer):
    identifier = serializers.CharField()
    password = serializers.CharField(write_only=True)


class ProfileUpdateSerializer(serializers.Serializer):
    first_name = serializers.CharField(max_length=150, required=False, allow_blank=True)
    last_name = serializers.CharField(max_length=150, required=False, allow_blank=True)
    email = serializers.EmailField(required=False)
    phone_number = serializers.CharField(max_length=32, required=False, allow_blank=True)
    age = serializers.IntegerField(required=False, min_value=0, max_value=120, allow_null=True)
    gender = serializers.ChoiceField(
        choices=[choice for choice, _ in UserProfile.Gender.choices],
        required=False,
        allow_blank=True,
    )
    address = serializers.CharField(max_length=255, required=False, allow_blank=True)
    emergency_contact_name = serializers.CharField(max_length=120, required=False, allow_blank=True)
    emergency_contact_phone = serializers.CharField(max_length=32, required=False, allow_blank=True)
    medical_history = serializers.JSONField(required=False)
    medical_profile = serializers.JSONField(required=False)
    preferred_language = serializers.CharField(max_length=16, required=False, allow_blank=True)

    def validate_email(self, value):
        email = value.strip().lower()
        user = self.context["request"].user
        if User.objects.exclude(pk=user.pk).filter(email__iexact=email).exists():
            raise serializers.ValidationError("Email is already in use.")
        return email

    def validate_medical_history(self, value):
        if value is None:
            return {}
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise serializers.ValidationError("medical_history must be valid JSON.") from exc
            if not isinstance(parsed, dict):
                raise serializers.ValidationError("medical_history must be a JSON object.")
            return parsed
        if not isinstance(value, dict):
            raise serializers.ValidationError("medical_history must be a JSON object.")
        return value

    def validate_medical_profile(self, value):
        if value is None:
            return {}
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise serializers.ValidationError("medical_profile must be valid JSON.") from exc
            if not isinstance(parsed, dict):
                raise serializers.ValidationError("medical_profile must be a JSON object.")
            return parsed
        if not isinstance(value, dict):
            raise serializers.ValidationError("medical_profile must be a JSON object.")
        return value


class ChatSessionSerializer(serializers.ModelSerializer):
    last_message = serializers.SerializerMethodField()
    message_count = serializers.SerializerMethodField()

    class Meta:
        model = ChatSession
        fields = ("id", "title", "created_at", "updated_at", "last_message", "message_count")

    def get_last_message(self, obj: ChatSession) -> str:
        last = obj.messages.order_by("-created_at").first()
        if not last:
            return ""
        return (last.content or "")[:140]

    def get_message_count(self, obj: ChatSession) -> int:
        return obj.messages.count()


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = ("id", "session", "role", "content", "metadata", "created_at")
        read_only_fields = ("id", "created_at", "session")


class HealthcareFacilitySerializer(serializers.ModelSerializer):
    class Meta:
        model = HealthcareFacility
        fields = (
            "id",
            "name",
            "facility_type",
            "specialization",
            "address",
            "phone_number",
            "latitude",
            "longitude",
            "is_emergency",
            "source",
            "created_at",
            "updated_at",
        )
        read_only_fields = ("id", "source", "created_at", "updated_at")


class FacilitySearchSerializer(serializers.Serializer):
    location_lat = serializers.FloatField(required=True)
    location_lng = serializers.FloatField(required=True)
    facility_type = serializers.ChoiceField(
        choices=[choice for choice, _ in HealthcareFacility.FacilityType.choices],
        required=False,
    )
    specialization = serializers.CharField(max_length=120, required=False, allow_blank=True)
    radius_km = serializers.IntegerField(required=False, min_value=1, max_value=50)


class AdminUserUpdateSerializer(serializers.Serializer):
    first_name = serializers.CharField(max_length=150, required=False, allow_blank=True)
    last_name = serializers.CharField(max_length=150, required=False, allow_blank=True)
    email = serializers.EmailField(required=False)
    is_active = serializers.BooleanField(required=False)
    is_staff = serializers.BooleanField(required=False)
