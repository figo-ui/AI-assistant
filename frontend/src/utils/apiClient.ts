import type {
  AnalysisResponse,
  AuthResponse,
  AuthTokens,
  AdminAnalytics,
  AdminFacility,
  ChatSession,
  ChatSessionPayload,
  EmergencyContact,
  Facility,
  Profile,
  SessionAnalyzeResponse,
} from "../types/analysis";

const viteEnv = (import.meta as unknown as { env?: Record<string, string | undefined> }).env;
const runtimeEnv = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env;
const API_BASE_URL =
  viteEnv?.VITE_HEALTHCARE_API_BASE_URL ||
  runtimeEnv?.REACT_APP_HEALTHCARE_API_BASE_URL ||
  "http://127.0.0.1:8000/api/v1";
const TOKENS_STORAGE_KEY = "healthcare_chatbot_tokens_v1";

function extractApiErrorMessage(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== "object") {
    return fallback;
  }
  const record = payload as Record<string, unknown>;

  if (typeof record.error === "string" && record.error.trim()) {
    return record.error;
  }
  if (typeof record.detail === "string" && record.detail.trim()) {
    return record.detail;
  }
  if (Array.isArray(record.non_field_errors) && record.non_field_errors.length) {
    return String(record.non_field_errors[0]);
  }

  for (const [key, value] of Object.entries(record)) {
    if (Array.isArray(value) && value.length) {
      return `${key}: ${String(value[0])}`;
    }
    if (typeof value === "string" && value.trim()) {
      return `${key}: ${value}`;
    }
  }
  return fallback;
}

export interface AnalyzeCasePayload {
  symptomText: string;
  symptomTags: string[];
  imageFile?: File | null;
  consentGiven: boolean;
  searchConsentGiven?: boolean;
  modelProfile?: "Clinical Balanced" | "Clinical Fast" | "Clinical Thorough";
  metadata?: Record<string, unknown>;
  facilityType?: string;
  specialization?: string;
  radiusKm?: number;
  location?: {
    lat: number;
    lng: number;
  } | null;
}

export interface ProfileUpdatePayload {
  first_name?: string;
  last_name?: string;
  email?: string;
  phone_number?: string;
  age?: number | null;
  gender?: string;
  address?: string;
  emergency_contact_name?: string;
  emergency_contact_phone?: string;
  medical_history?: Record<string, unknown>;
  medical_profile?: Record<string, unknown>;
  preferred_language?: string;
}

export interface FacilityPayload {
  name: string;
  facility_type: string;
  specialization?: string;
  address?: string;
  phone_number?: string;
  latitude: number;
  longitude: number;
  is_emergency?: boolean;
}

export interface RegisterPayload {
  username?: string;
  email: string;
  first_name?: string;
  last_name?: string;
  password: string;
  phone_number?: string;
}

export interface LoginPayload {
  identifier: string;
  password: string;
}

function getStoredTokens(): AuthTokens | null {
  const raw = localStorage.getItem(TOKENS_STORAGE_KEY);
  if (!raw) {
    return null;
  }
  try {
    return JSON.parse(raw) as AuthTokens;
  } catch {
    return null;
  }
}

export function setStoredTokens(tokens: AuthTokens | null): void {
  if (!tokens) {
    localStorage.removeItem(TOKENS_STORAGE_KEY);
    return;
  }
  localStorage.setItem(TOKENS_STORAGE_KEY, JSON.stringify(tokens));
}

async function refreshAccessToken(): Promise<boolean> {
  const tokens = getStoredTokens();
  try {
    const response = await fetch(`${API_BASE_URL}/auth/refresh/`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(tokens?.refresh ? { refresh: tokens.refresh } : {}),
    });
    if (!response.ok) {
      setStoredTokens(null);
      return false;
    }
    return true;
  } catch {
    setStoredTokens(null);
    return false;
  }
}

async function requestJSON<T>(
  path: string,
  init: RequestInit = {},
  withAuth = true,
  allowRetry = true,
): Promise<T> {
  const headers = new Headers(init.headers || {});
  if (!headers.has("Content-Type") && !(init.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  if (withAuth) {
    const tokens = getStoredTokens();
    if (tokens?.access) {
      headers.set("Authorization", `Bearer ${tokens.access}`);
    }
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { ...init, headers, credentials: "include" });
  } catch {
    throw new Error(`Failed to reach backend at ${API_BASE_URL}.`);
  }

  if (response.status === 401 && withAuth && allowRetry) {
    const refreshed = await refreshAccessToken();
    if (refreshed) {
      return requestJSON<T>(path, init, withAuth, false);
    }
  }

  const raw = await response.text();
  let payload: Record<string, unknown> = {};
  if (raw) {
    try {
      payload = JSON.parse(raw) as Record<string, unknown>;
    } catch {
      payload = { error: raw };
    }
  }
  if (!response.ok) {
    throw new Error(extractApiErrorMessage(payload, `Request failed (${response.status}).`));
  }
  return payload as unknown as T;
}

export async function register(payload: RegisterPayload): Promise<AuthResponse> {
  const response = await requestJSON<AuthResponse>(
    "/auth/register/",
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
    false,
  );
  if (response.tokens) {
    setStoredTokens(response.tokens);
  }
  return response;
}

export async function login(payload: LoginPayload): Promise<AuthResponse> {
  const response = await requestJSON<AuthResponse>(
    "/auth/login/",
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
    false,
  );
  if (response.tokens) {
    setStoredTokens(response.tokens);
  }
  return response;
}

export async function logout(): Promise<void> {
  const tokens = getStoredTokens();
  if (tokens?.refresh) {
    await requestJSON<{ status: string }>(
      "/auth/logout/",
      {
        method: "POST",
        body: JSON.stringify({ refresh: tokens.refresh }),
      },
      true,
      false,
    ).catch(() => undefined);
  }
  setStoredTokens(null);
}

export async function getProfile(): Promise<Profile> {
  return requestJSON<Profile>("/profile/");
}

export async function updateProfile(payload: ProfileUpdatePayload): Promise<Profile> {
  return requestJSON<Profile>("/profile/", {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export async function listSessions(query: {
  q?: string;
  date_from?: string;
  date_to?: string;
}): Promise<ChatSession[]> {
  const params = new URLSearchParams();
  if (query.q) params.set("q", query.q);
  if (query.date_from) params.set("date_from", query.date_from);
  if (query.date_to) params.set("date_to", query.date_to);
  const suffix = params.toString() ? `?${params.toString()}` : "";
  return requestJSON<ChatSession[]>(`/chat/sessions/${suffix}`);
}

export async function createSession(title?: string): Promise<ChatSession> {
  return requestJSON<ChatSession>("/chat/sessions/", {
    method: "POST",
    body: JSON.stringify({ title: title || "Health Consultation" }),
  });
}

export async function getSessionMessages(sessionId: number): Promise<ChatSessionPayload> {
  return requestJSON<ChatSessionPayload>(`/chat/sessions/${sessionId}/messages/`);
}

export async function analyzeInSession(
  sessionId: number,
  payload: AnalyzeCasePayload,
): Promise<SessionAnalyzeResponse> {
  const body = new FormData();
  body.append("symptom_text", payload.symptomText);
  body.append("consent_given", payload.consentGiven ? "true" : "false");
  body.append("search_consent_given", payload.searchConsentGiven ? "true" : "false");
  if (payload.modelProfile) body.append("model_profile", payload.modelProfile);
  if (payload.symptomTags.length) {
    payload.symptomTags.forEach((tag) => body.append("symptom_tags", tag));
  }
  if (payload.imageFile) body.append("image", payload.imageFile);
  if (payload.location) {
    body.append("location_lat", String(payload.location.lat));
    body.append("location_lng", String(payload.location.lng));
  }
  if (payload.metadata) body.append("metadata", JSON.stringify(payload.metadata));
  if (payload.facilityType) body.append("facility_type", payload.facilityType);
  if (payload.specialization) body.append("specialization", payload.specialization);
  if (payload.radiusKm) body.append("search_radius_km", String(payload.radiusKm));

  let response = await fetch(`${API_BASE_URL}/chat/sessions/${sessionId}/analyze/`, {
    method: "POST",
    body,
    credentials: "include",
  });

  if (response.status === 401) {
    const refreshed = await refreshAccessToken();
    if (refreshed) {
      response = await fetch(`${API_BASE_URL}/chat/sessions/${sessionId}/analyze/`, {
        method: "POST",
        body,
        credentials: "include",
      });
    }
  }

  const raw = await response.text();
  let parsed: Record<string, unknown> = {};
  if (raw) {
    try {
      parsed = JSON.parse(raw) as Record<string, unknown>;
    } catch {
      parsed = { error: raw };
    }
  }
  if (!response.ok) {
    throw new Error(extractApiErrorMessage(parsed, `Analysis request failed (${response.status}).`));
  }
  return parsed as unknown as SessionAnalyzeResponse;
}

export async function getChatHistory(query: {
  q?: string;
  date_from?: string;
  date_to?: string;
}): Promise<unknown[]> {
  const params = new URLSearchParams();
  if (query.q) params.set("q", query.q);
  if (query.date_from) params.set("date_from", query.date_from);
  if (query.date_to) params.set("date_to", query.date_to);
  const suffix = params.toString() ? `?${params.toString()}` : "";
  return requestJSON<unknown[]>(`/chat/history/${suffix}`);
}

export async function getNearbyFacilities(params: {
  location_lat: number;
  location_lng: number;
  facility_type?: string;
  specialization?: string;
  radius_km?: number;
}): Promise<Facility[]> {
  const query = new URLSearchParams({
    location_lat: String(params.location_lat),
    location_lng: String(params.location_lng),
  });
  if (params.facility_type) query.set("facility_type", params.facility_type);
  if (params.specialization) query.set("specialization", params.specialization);
  if (params.radius_km) query.set("radius_km", String(params.radius_km));

  const payload = await requestJSON<{ facilities: Facility[] }>(`/location/nearby/?${query.toString()}`, {}, false);
  return payload.facilities;
}

export async function getEmergencyContacts(countryCode?: string): Promise<EmergencyContact[]> {
  const query = new URLSearchParams();
  if (countryCode) {
    query.set("country_code", countryCode);
  }
  const suffix = query.toString() ? `?${query.toString()}` : "";
  const payload = await requestJSON<{ contacts: EmergencyContact[] }>(`/location/emergency/${suffix}`, {}, false);
  return payload.contacts;
}

export async function exportProfileJSON(): Promise<Record<string, unknown>> {
  return requestJSON<Record<string, unknown>>("/export/profile/");
}

export async function exportChatHistoryJSON(sessionId?: number): Promise<Record<string, unknown>> {
  const query = new URLSearchParams({ format: "json" });
  if (sessionId) query.set("session_id", String(sessionId));
  return requestJSON<Record<string, unknown>>(`/chat/export/?${query.toString()}`);
}

export async function exportChatHistoryCSV(sessionId?: number): Promise<string> {
  const query = new URLSearchParams({ format: "csv" });
  if (sessionId) query.set("session_id", String(sessionId));
  const response = await fetch(`${API_BASE_URL}/chat/export/?${query.toString()}`, {
    method: "GET",
    credentials: "include",
  });
  if (!response.ok) {
    throw new Error("Unable to export chat history as CSV.");
  }
  return response.text();
}

export async function verifyEmail(token: string): Promise<{ status: string; email?: string }> {
  return requestJSON(`/auth/verify-email/?token=${encodeURIComponent(token)}`, {}, false);
}

export async function resendVerificationEmail(): Promise<{ status: string }> {
  return requestJSON("/auth/resend-verification/", { method: "POST" });
}

export async function requestPasswordReset(email: string): Promise<{ status: string }> {
  return requestJSON("/auth/password-reset/", { method: "POST", body: JSON.stringify({ email }) }, false);
}

export async function confirmPasswordReset(token: string, newPassword: string): Promise<{ status: string }> {
  return requestJSON(
    "/auth/password-reset/confirm/",
    { method: "POST", body: JSON.stringify({ token, new_password: newPassword }) },
    false,
  );
}

export async function getAdminAnalytics(): Promise<AdminAnalytics> {

export async function listAdminUsers(): Promise<{ count: number; results: User[] }> {
  return requestJSON<{ count: number; results: User[] }>("/admin/users/");
}

export async function updateAdminUser(
  userId: number,
  payload: { first_name?: string; last_name?: string; email?: string; is_active?: boolean; is_staff?: boolean },
): Promise<User> {
  return requestJSON<User>(`/admin/users/${userId}/`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export interface AuditLogEntry {
  id: number;
  actor: string | null;
  action: string;
  target_type: string;
  target_id: string;
  metadata: Record<string, unknown>;
  created_at: string;
}

export async function getAdminAuditLog(params?: {
  page?: number;
  page_size?: number;
  action?: string;
}): Promise<{ total: number; page: number; page_size: number; results: AuditLogEntry[] }> {
  const query = new URLSearchParams();
  if (params?.page) query.set("page", String(params.page));
  if (params?.page_size) query.set("page_size", String(params.page_size));
  if (params?.action) query.set("action", params.action);
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return requestJSON(`/admin/audit-log/${suffix}`);
}

export async function triggerAdminRetrain(): Promise<{ status: string; message: string }> {
  return requestJSON("/admin/config/", {
    method: "POST",
    body: JSON.stringify({ action: "retrain_text_model" }),
  });
}

export async function getAdminModelMetrics(): Promise<Record<string, unknown>> {
  return requestJSON("/admin/model-metrics/");
}

export async function getAdminDialogueTemplates(): Promise<{ templates: Record<string, string[]> }> {
  return requestJSON("/admin/dialogue-templates/");
}

export async function updateAdminDialogueTemplate(
  intent: string,
  templates: string[],
): Promise<{ status: string; intent: string; templates: string[] }> {
  return requestJSON("/admin/dialogue-templates/", {
    method: "POST",
    body: JSON.stringify({ intent, templates }),
  });
}

export function subscribeToAnalysisSSE(
  caseId: number,
  token: string,
  onMessage: (data: Record<string, unknown>) => void,
  onDone: () => void,
): () => void {
  const url = `${API_BASE_URL}/analyze/${caseId}/stream/?token=${encodeURIComponent(token)}`;
  const source = new EventSource(url, { withCredentials: true });
  source.onmessage = (event) => {
    try {
      const parsed = JSON.parse(event.data as string) as Record<string, unknown>;
      onMessage(parsed);
      const s = parsed.status as string;
      if (s === "completed" || s === "failed" || s === "timeout" || s === "unknown") {
        source.close();
        onDone();
      }
    } catch {
      // ignore parse errors
    }
  };
  source.onerror = () => {
    source.close();
    onDone();
  };
  return () => source.close();
}

export async function listAdminFacilities(): Promise<AdminFacility[]> {
  return requestJSON<AdminFacility[]>("/admin/facilities/");
}

export async function createAdminFacility(payload: FacilityPayload): Promise<AdminFacility> {
  return requestJSON<AdminFacility>("/admin/facilities/", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function updateAdminFacility(
  facilityId: number,
  payload: Partial<FacilityPayload>,
): Promise<AdminFacility> {
  return requestJSON<AdminFacility>(`/admin/facilities/${facilityId}/`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export async function deleteAdminFacility(facilityId: number): Promise<void> {
  await requestJSON(`/admin/facilities/${facilityId}/`, {
    method: "DELETE",
  });
}

export async function analyzeCase(payload: AnalyzeCasePayload): Promise<AnalysisResponse> {
  const body = new FormData();
  body.append("symptom_text", payload.symptomText);
  body.append("consent_given", payload.consentGiven ? "true" : "false");
  body.append("search_consent_given", payload.searchConsentGiven ? "true" : "false");
  if (payload.modelProfile) body.append("model_profile", payload.modelProfile);
  payload.symptomTags.forEach((tag) => body.append("symptom_tags", tag));
  if (payload.imageFile) body.append("image", payload.imageFile);
  if (payload.location) {
    body.append("location_lat", String(payload.location.lat));
    body.append("location_lng", String(payload.location.lng));
  }
  if (payload.metadata) body.append("metadata", JSON.stringify(payload.metadata));
  if (payload.facilityType) body.append("facility_type", payload.facilityType);
  if (payload.specialization) body.append("specialization", payload.specialization);
  if (payload.radiusKm) body.append("search_radius_km", String(payload.radiusKm));

  const response = await fetch(`${API_BASE_URL}/analyze/`, {
    method: "POST",
    body,
    credentials: "include",
  });
  const raw = await response.text();
  let parsed: Record<string, unknown> = {};
  if (raw) {
    try {
      parsed = JSON.parse(raw) as Record<string, unknown>;
    } catch {
      parsed = { error: raw };
    }
  }
  if (!response.ok) {
    throw new Error(extractApiErrorMessage(parsed, "Analysis request failed."));
  }
  return parsed as unknown as AnalysisResponse;
}

export function getCurrentLocation(timeoutMs = 5000): Promise<{ lat: number; lng: number }> {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error("Geolocation is not supported by this browser."));
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (position) => {
        resolve({
          lat: position.coords.latitude,
          lng: position.coords.longitude,
        });
      },
      () => reject(new Error("Unable to access your location.")),
      { timeout: timeoutMs, maximumAge: 120000, enableHighAccuracy: false },
    );
  });
}
