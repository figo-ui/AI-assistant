import math
import hashlib
import json as _json
from functools import lru_cache
from typing import Dict, List, Optional

import requests
from django.conf import settings
from django.core.cache import cache
from django.db.models import Q

from ..models import HealthcareFacility

# Cache TTL for Google Places results (seconds)
_PLACES_CACHE_TTL = int(getattr(settings, "PLACES_CACHE_TTL_SECONDS", 3600))


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lng / 2) ** 2
    )
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def lookup_nearby_facilities(
    location_lat: Optional[float],
    location_lng: Optional[float],
    facility_type: str = "hospital",
    specialization: str = "",
    radius_km: int = 5,
    limit: int = 5,
) -> List[Dict]:
    if location_lat is None or location_lng is None:
        return []

    api_key = settings.GOOGLE_MAPS_API_KEY
    radius_km = max(1, min(50, int(radius_km or 5)))
    specialization = (specialization or "").strip()
    facility_type = (facility_type or "hospital").strip().lower()

    if api_key:
        google_results = _lookup_google_places(
            location_lat=location_lat,
            location_lng=location_lng,
            facility_type=facility_type,
            specialization=specialization,
            radius_km=radius_km,
            limit=limit,
            api_key=api_key,
        )
        if google_results:
            return google_results

    return _lookup_local_facilities(
        location_lat=location_lat,
        location_lng=location_lng,
        facility_type=facility_type,
        specialization=specialization,
        radius_km=radius_km,
        limit=limit,
    )


def _places_cache_key(location_lat: float, location_lng: float, facility_type: str, specialization: str, radius_km: int) -> str:
    raw = f"{location_lat:.4f}:{location_lng:.4f}:{facility_type}:{specialization}:{radius_km}"
    return "places:" + hashlib.md5(raw.encode()).hexdigest()


def _lookup_google_places(
    *,
    location_lat: float,
    location_lng: float,
    facility_type: str,
    specialization: str,
    radius_km: int,
    limit: int,
    api_key: str,
) -> List[Dict]:
    # REQ-4: Redis cache — avoid repeated Places API calls for same area
    cache_key = _places_cache_key(location_lat, location_lng, facility_type, specialization, radius_km)
    try:
        cached = cache.get(cache_key)
        if cached is not None:
            return _json.loads(cached)[:limit]
    except Exception:
        pass

    endpoint = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    google_type = "hospital" if facility_type in {"hospital", "emergency"} else facility_type
    params = {
        "key": api_key,
        "location": f"{location_lat},{location_lng}",
        "radius": radius_km * 1000,
        "type": google_type,
    }
    if specialization:
        params["keyword"] = specialization

    try:
        response = requests.get(endpoint, params=params, timeout=6)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    facilities: List[Dict] = []
    for item in payload.get("results", []):
        geometry = item.get("geometry", {}).get("location", {})
        lat = geometry.get("lat")
        lng = geometry.get("lng")
        if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
            continue

        distance_km = round(_haversine_km(location_lat, location_lng, lat, lng), 2)
        if distance_km > radius_km:
            continue

        place_id = item.get("place_id", "")
        opening_hours = item.get("opening_hours", {})
        facilities.append(
            {
                "provider_name": item.get("name", "Unknown provider"),
                "address": item.get("vicinity", ""),
                "place_id": place_id,
                "distance_km": distance_km,
                "rating": item.get("rating"),
                "open_now": opening_hours.get("open_now"),
                "maps_url": f"https://www.google.com/maps/place/?q=place_id:{place_id}" if place_id else "",
                "source": "google_places",
                "phone_number": "",
                "facility_type": facility_type,
                "specialization": specialization,
                "is_emergency": facility_type == "emergency",
            }
        )

    facilities.sort(key=lambda x: (x.get("distance_km") is None, x.get("distance_km", 9999.0)))
    result = facilities[:limit]

    # Store in cache
    try:
        cache.set(cache_key, _json.dumps(result), timeout=_PLACES_CACHE_TTL)
    except Exception:
        pass

    return result


def _lookup_local_facilities(
    *,
    location_lat: float,
    location_lng: float,
    facility_type: str,
    specialization: str,
    radius_km: int,
    limit: int,
) -> List[Dict]:
    queryset = HealthcareFacility.objects.all()
    if facility_type:
        if facility_type == "emergency":
            queryset = queryset.filter(Q(is_emergency=True) | Q(facility_type=HealthcareFacility.FacilityType.EMERGENCY))
        else:
            queryset = queryset.filter(facility_type=facility_type)
    if specialization:
        queryset = queryset.filter(specialization__icontains=specialization)

    facilities: List[Dict] = []
    for facility in queryset:
        distance = _haversine_km(location_lat, location_lng, facility.latitude, facility.longitude)
        if distance <= radius_km:
            facilities.append(
                {
                    "provider_name": facility.name,
                    "address": facility.address,
                    "place_id": "",
                    "distance_km": round(distance, 2),
                    "rating": None,
                    "maps_url": (
                        f"https://www.google.com/maps/search/?api=1&query={facility.latitude},{facility.longitude}"
                    ),
                    "source": "local_registry",
                    "phone_number": facility.phone_number,
                    "facility_type": facility.facility_type,
                    "specialization": facility.specialization,
                    "is_emergency": facility.is_emergency,
                }
            )

    facilities.sort(key=lambda x: x.get("distance_km", 9999.0))
    return facilities[:limit]


EMERGENCY_CONTACTS_BY_LOCALE: Dict[str, List[Dict[str, str]]] = {
    "US": [
        {"name": "Emergency Services (Police/Fire/Medical)", "phone_number": "911", "region": "United States"},
        {"name": "Poison Control", "phone_number": "1-800-222-1222", "region": "United States"},
    ],
    "ET": [
        {"name": "Ambulance", "phone_number": "907", "region": "Ethiopia"},
        {"name": "Police", "phone_number": "991", "region": "Ethiopia"},
        {"name": "Fire", "phone_number": "939", "region": "Ethiopia"},
    ],
    "KE": [
        {"name": "Emergency Services", "phone_number": "999", "region": "Kenya"},
        {"name": "Ambulance (AMREF)", "phone_number": "0800 723 253", "region": "Kenya"},
    ],
    "GB": [
        {"name": "Emergency Services", "phone_number": "999", "region": "United Kingdom"},
        {"name": "NHS Non-Emergency", "phone_number": "111", "region": "United Kingdom"},
    ],
    "IN": [
        {"name": "Ambulance", "phone_number": "108", "region": "India"},
        {"name": "Police", "phone_number": "100", "region": "India"},
        {"name": "National Emergency", "phone_number": "112", "region": "India"},
    ],
    "NG": [
        {"name": "Emergency Services", "phone_number": "112", "region": "Nigeria"},
        {"name": "Ambulance (Lagos)", "phone_number": "767", "region": "Nigeria"},
    ],
    "ZA": [
        {"name": "Emergency Services", "phone_number": "10111", "region": "South Africa"},
        {"name": "Ambulance", "phone_number": "10177", "region": "South Africa"},
    ],
    "CA": [
        {"name": "Emergency Services", "phone_number": "911", "region": "Canada"},
        {"name": "Poison Control", "phone_number": "1-800-268-9017", "region": "Canada"},
    ],
    "AU": [
        {"name": "Emergency Services", "phone_number": "000", "region": "Australia"},
        {"name": "Non-Emergency Medical", "phone_number": "1800 022 222", "region": "Australia"},
    ],
    "DE": [
        {"name": "Emergency Services", "phone_number": "112", "region": "Germany"},
        {"name": "Police", "phone_number": "110", "region": "Germany"},
    ],
    "EU": [
        {"name": "European Emergency Number", "phone_number": "112", "region": "European Union"},
    ],
}

# Default fallback contacts shown when country is unknown
_DEFAULT_CONTACTS = [
    {"name": "International Emergency (most countries)", "phone_number": "112", "region": "International"},
    {"name": "US Emergency", "phone_number": "911", "region": "United States"},
    {"name": "Ethiopia Ambulance", "phone_number": "907", "region": "Ethiopia"},
]


def emergency_contacts(country_code: str = "") -> List[Dict[str, str]]:
    normalized = str(country_code or "").strip().upper()
    if normalized in EMERGENCY_CONTACTS_BY_LOCALE:
        return EMERGENCY_CONTACTS_BY_LOCALE[normalized]
    return _DEFAULT_CONTACTS
