export interface ConditionPrediction {
  condition: string;
  probability: number;
}

export interface Facility {
  id?: number;
  provider_name: string;
  address: string;
  place_id: string;
  distance_km: number | null;
  rating: number | null;
  open_now?: boolean | null;
  maps_url: string;
  source?: string;
  phone_number?: string;
  facility_type?: string;
  specialization?: string;
  is_emergency?: boolean;
  latitude?: number;
  longitude?: number;
}

export interface AnalysisResponse {
  case_id: number;
  created_at: string;
  probable_conditions: ConditionPrediction[];
  risk_score: number;
  risk_level: "Low" | "Medium" | "High";
  confidence_band: string;
  recommendation_text: string;
  prevention_advice: string[];
  disclaimer_text: string;
  risk_factors: string[];
  red_flags: string[];
  needs_urgent_care: boolean;
  emergency_auto_triggered?: boolean;
  response_format_sections: string[];
  clinical_report: Record<string, unknown>;
  modality_predictions: {
    text: ConditionPrediction[];
    image: ConditionPrediction[];
  };
  nearby_facilities: Facility[];
  model_versions: {
    text: string;
    image: string;
    fusion: string;
  };
  latency_ms: number;
}

export interface User {
  id: number;
  username: string;
  email: string;
  first_name: string;
  last_name: string;
  is_active: boolean;
  is_staff: boolean;
}

export interface Profile {
  user: User;
  phone_number: string;
  age: number | null;
  gender: string;
  address: string;
  emergency_contact_name: string;
  emergency_contact_phone: string;
  medical_history: Record<string, unknown>;
  medical_profile?: Record<string, unknown>;
  preferred_language: string;
  email_verified?: boolean;
  created_at: string;
  updated_at: string;
}

export interface AuthTokens {
  access: string;
  refresh: string;
}

export interface AuthResponse {
  user: User;
  profile: Profile;
  tokens?: AuthTokens;
}

export interface ChatSession {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
  last_message: string;
  message_count: number;
}

export interface ChatMessage {
  id: number;
  session: number;
  role: "user" | "assistant" | "system";
  content: string;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface ChatSessionPayload {
  session: ChatSession;
  messages: ChatMessage[];
}

export interface SessionAnalyzeResponse {
  session: ChatSession;
  user_message: ChatMessage;
  assistant_message: ChatMessage;
  analysis: AnalysisResponse;
}

export interface EmergencyContact {
  name: string;
  phone_number: string;
  region: string;
}

export interface RiskBreakdownItem {
  risk__risk_level: string;
  count: number;
}

export interface AdminAnalytics {
  users_total: number;
  users_active: number;
  chat_sessions_total: number;
  chat_messages_total: number;
  cases_total: number;
  cases_completed: number;
  facilities_total: number;
  cases_today?: number;
  risk_breakdown: RiskBreakdownItem[];
}

export interface AdminFacility {
  id: number;
  name: string;
  facility_type: string;
  specialization: string;
  address: string;
  phone_number: string;
  latitude: number;
  longitude: number;
  is_emergency: boolean;
  source?: string;
  created_at?: string;
  updated_at?: string;
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
