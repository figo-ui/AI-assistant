import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import type React from "react";

import { AuthScreen } from "./features/auth/AuthScreen";
import { ChatPane } from "./features/chat/ChatPane";
import { latestAnalysis, type LocalAttachment } from "./features/chat/chatUtils";
import { FacilitiesPanel } from "./features/facilities/FacilitiesPanel";
import { GuidancePanel } from "./features/guidance/GuidancePanel";
import { AppSidebar } from "./features/layout/AppSidebar";
import { MainHeader } from "./features/layout/MainHeader";
import { AdminDashboard } from "./features/admin/AdminDashboard";
import { ProfilePanel } from "./features/profile/ProfilePanel";
import { SettingsModal } from "./features/settings/SettingsModal";
import type {
  AnalysisResponse,
  ChatMessage,
  ChatSession,
  EmergencyContact,
  Facility,
  Profile,
  User,
} from "./types/analysis";
import {
  analyzeInSession,
  createSession,
  exportChatHistoryCSV,
  exportChatHistoryJSON,
  exportProfileJSON,
  getCurrentLocation,
  getEmergencyContacts,
  getNearbyFacilities,
  getProfile,
  getSessionMessages,
  listSessions,
  login,
  logout,
  register,
  requestPasswordReset,
  confirmPasswordReset,
  resendVerificationEmail,
  setStoredTokens,
  subscribeToAnalysisSSE,
  updateProfile,
} from "./utils/apiClient";

type ThemeMode = "light" | "dark";
type AppTab = "chat" | "guidance" | "facilities" | "profile" | "admin";

function downloadText(filename: string, content: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export default function App() {
  const [booting, setBooting] = useState(true);
  const [user, setUser] = useState<User | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [activeTab, setActiveTab] = useState<AppTab>("chat");
  const [theme, setTheme] = useState<ThemeMode>(() => {
    if (typeof window === "undefined") return "light";
    return localStorage.getItem("healthcare_ui_theme") === "dark" ? "dark" : "light";
  });

  const [mode, setMode] = useState<"login" | "register" | "forgot" | "reset">("login");
  const [identifier, setIdentifier] = useState("");
  const [password, setPassword] = useState("");
  const [regEmail, setRegEmail] = useState("");
  const [regPassword, setRegPassword] = useState("");
  const [regFirstName, setRegFirstName] = useState("");
  const [regLastName, setRegLastName] = useState("");
  const [regPhone, setRegPhone] = useState("");
  const [forgotEmail, setForgotEmail] = useState("");
  const [resetToken, setResetToken] = useState("");
  const [resetPassword, setResetPassword] = useState("");
  const [resetPasswordConfirm, setResetPasswordConfirm] = useState("");

  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [facilities, setFacilities] = useState<Facility[]>([]);
  const [contacts, setContacts] = useState<EmergencyContact[]>([]);

  const [symptomText, setSymptomText] = useState("");
  const [symptomTagsText, setSymptomTagsText] = useState("");
  const [consentGiven, setConsentGiven] = useState(true);
  const [searchConsentGiven, setSearchConsentGiven] = useState(false);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState("");
  const [facilityType, setFacilityType] = useState("hospital");
  const [specialization, setSpecialization] = useState("");
  const [radiusKm, setRadiusKm] = useState(5);
  const [locationLat, setLocationLat] = useState("");
  const [locationLng, setLocationLng] = useState("");

  const [profileFirstName, setProfileFirstName] = useState("");
  const [profileLastName, setProfileLastName] = useState("");
  const [profileEmail, setProfileEmail] = useState("");
  const [profilePhone, setProfilePhone] = useState("");
  const [profileAge, setProfileAge] = useState("");
  const [profileGender, setProfileGender] = useState("");
  const [profileEmergencyName, setProfileEmergencyName] = useState("");
  const [profileEmergencyPhone, setProfileEmergencyPhone] = useState("");
  const [profileEmergencyEmail, setProfileEmergencyEmail] = useState("");
  const [profileConditionsText, setProfileConditionsText] = useState("");
  const [profileAllergiesText, setProfileAllergiesText] = useState("");
  const [profileMedicationsText, setProfileMedicationsText] = useState("");
  const [profileComorbiditiesText, setProfileComorbiditiesText] = useState("");
  const [profilePregnancyStatus, setProfilePregnancyStatus] = useState("");

  const [showComposerDetails, setShowComposerDetails] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState("Clinical Balanced");
  const [feedbackByMessage, setFeedbackByMessage] = useState<Record<number, "up" | "down">>({});
  const [hiddenMessageIds, setHiddenMessageIds] = useState<number[]>([]);
  const [messageAttachments, setMessageAttachments] = useState<Record<number, LocalAttachment>>({});
  const [streamingMessageId, setStreamingMessageId] = useState<number | null>(null);
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);

  const chatFeedRef = useRef<HTMLDivElement | null>(null);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);

  const activeSession = sessions.find((entry) => entry.id === activeSessionId) || null;
  const recentAssistantMessages = messages.filter((entry) => entry.role === "assistant").slice(-5).reverse();
  const hiddenSet = useMemo(() => new Set(hiddenMessageIds), [hiddenMessageIds]);
  const visibleMessages = useMemo(() => messages.filter((entry) => !hiddenSet.has(entry.id)), [messages, hiddenSet]);

  function parseCommaList(value: string): string[] {
    return value
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
  }

  function listToText(value: unknown): string {
    if (!Array.isArray(value)) return "";
    return value.map((item) => String(item).trim()).filter(Boolean).join(", ");
  }

  function hydrateProfileForm(nextProfile: Profile): void {
    setProfileFirstName(nextProfile.user.first_name || "");
    setProfileLastName(nextProfile.user.last_name || "");
    setProfileEmail(nextProfile.user.email || "");
    setProfilePhone(nextProfile.phone_number || "");
    setProfileAge(nextProfile.age === null ? "" : String(nextProfile.age));
    setProfileGender(nextProfile.gender || "");
    setProfileEmergencyName(nextProfile.emergency_contact_name || "");
    setProfileEmergencyPhone(nextProfile.emergency_contact_phone || "");
    const source = nextProfile.medical_profile || nextProfile.medical_history || {};
    const record = source as Record<string, unknown>;
    setProfileConditionsText(listToText(record.conditions));
    setProfileAllergiesText(listToText(record.allergies));
    setProfileMedicationsText(listToText(record.medications));
    setProfileComorbiditiesText(listToText(record.comorbidities));
    setProfilePregnancyStatus(String(record.pregnancy_status || ""));
    setProfileEmergencyEmail(String(record.emergency_contact_email || ""));
  }

  async function refreshSessions(): Promise<void> {
    const listed = await listSessions({});
    setSessions(listed);
    if (!listed.length) {
      setActiveSessionId(null);
      setMessages([]);
      setAnalysis(null);
      return;
    }
    const targetId = activeSessionId && listed.some((item) => item.id === activeSessionId) ? activeSessionId : listed[0].id;
    setActiveSessionId(targetId);
    const payload = await getSessionMessages(targetId);
    setMessages(payload.messages);
    const recent = latestAnalysis(payload.messages);
    setAnalysis(recent);
    setFacilities(recent?.nearby_facilities || []);
  }

  useEffect(() => {
    const bootstrap = async () => {
      try {
        const nextProfile = await getProfile();
        setUser(nextProfile.user);
        setProfile(nextProfile);
        hydrateProfileForm(nextProfile);
        await refreshSessions();
      } catch {
        setStoredTokens(null);
      }
      try {
        setContacts(await getEmergencyContacts());
      } catch {
        // ignore
      }
      setBooting(false);
    };
    void bootstrap();
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("healthcare_ui_theme", theme);
  }, [theme]);

  useEffect(() => {
    if (!error && !notice) return;
    const timer = window.setTimeout(() => {
      setError("");
      setNotice("");
    }, 4500);
    return () => window.clearTimeout(timer);
  }, [error, notice]);

  useEffect(() => {
    const node = composerRef.current;
    if (!node) return;
    node.style.height = "0px";
    node.style.height = `${Math.min(node.scrollHeight, 220)}px`;
  }, [symptomText]);

  useEffect(() => {
    if (!streamingMessageId) return;
    const timer = window.setTimeout(() => setStreamingMessageId(null), 1400);
    return () => window.clearTimeout(timer);
  }, [streamingMessageId]);

  useEffect(() => {
    const node = chatFeedRef.current;
    if (!node || activeTab !== "chat") return;
    const onScroll = () => {
      const hiddenDistance = node.scrollHeight - node.scrollTop - node.clientHeight;
      setShowScrollToBottom(hiddenDistance > 220);
    };
    node.addEventListener("scroll", onScroll);
    onScroll();
    return () => node.removeEventListener("scroll", onScroll);
  }, [activeTab, visibleMessages.length, isTyping]);

  useEffect(() => {
    if (activeTab !== "chat") return;
    const node = chatFeedRef.current;
    if (!node) return;
    const hiddenDistance = node.scrollHeight - node.scrollTop - node.clientHeight;
    if (hiddenDistance < 180 || isTyping) {
      requestAnimationFrame(() => {
        node.scrollTo({ top: node.scrollHeight, behavior: "smooth" });
      });
    }
  }, [visibleMessages.length, isTyping, activeTab]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setActiveTab("chat");
        composerRef.current?.focus();
      }
      if (event.key === "Escape") {
        setMobileSidebarOpen(false);
        setSettingsOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => {
    return () => {
      if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
      Object.values(messageAttachments).forEach((item) => URL.revokeObjectURL(item.url));
    };
  }, [imagePreviewUrl, messageAttachments]);

  async function handleLogin(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const response = await login({ identifier, password });
      setUser(response.user);
      setProfile(response.profile);
      hydrateProfileForm(response.profile);
      await refreshSessions();
      setPassword("");
      setNotice("Logged in.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleRegister(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const response = await register({
        email: regEmail,
        password: regPassword,
        first_name: regFirstName,
        last_name: regLastName,
        phone_number: regPhone,
      });
      setUser(response.user);
      setProfile(response.profile);
      hydrateProfileForm(response.profile);
      await refreshSessions();
      setRegPassword("");
      setNotice("Account created.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registration failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleForgotPassword(event: React.FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await requestPasswordReset(forgotEmail);
      setNotice("If that email is registered, a reset link has been sent. Check your inbox.");
      setForgotEmail("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to send reset email.");
    } finally {
      setBusy(false);
    }
  }

  async function handleResetPassword(event: React.FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (resetPassword !== resetPasswordConfirm) {
      setError("Passwords do not match.");
      return;
    }
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await confirmPasswordReset(resetToken, resetPassword);
      setNotice("Password reset successfully. You can now log in.");
      setResetToken("");
      setResetPassword("");
      setResetPasswordConfirm("");
      setMode("login");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Password reset failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleLogout(): Promise<void> {
    setBusy(true);
    try {
      await logout();
    } finally {
      setUser(null);
      setProfile(null);
      setSessions([]);
      setMessages([]);
      setAnalysis(null);
      setFacilities([]);
      setBusy(false);
    }
  }

  async function handleCreateSession(): Promise<void> {
    setBusy(true);
    setError("");
    try {
      const session = await createSession();
      setSessions((prev) => [session, ...prev]);
      setActiveSessionId(session.id);
      setMessages([]);
      setAnalysis(null);
      setNotice("Session created.");
      setActiveTab("chat");
      setMobileSidebarOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to create session.");
    } finally {
      setBusy(false);
    }
  }

  async function handleSelectSession(sessionId: number): Promise<void> {
    setBusy(true);
    setError("");
    try {
      const payload = await getSessionMessages(sessionId);
      setActiveSessionId(sessionId);
      setMessages(payload.messages);
      const recent = latestAnalysis(payload.messages);
      setAnalysis(recent);
      setFacilities(recent?.nearby_facilities || []);
      setActiveTab("chat");
      setMobileSidebarOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to open session.");
    } finally {
      setBusy(false);
    }
  }

  async function handleDetectLocation(): Promise<void> {
    setError("");
    try {
      const location = await getCurrentLocation();
      setLocationLat(String(location.lat));
      setLocationLng(String(location.lng));
      setNotice("Location detected.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to detect location.");
    }
  }

  async function handleFacilitySearch(emergencyOnly = false): Promise<void> {
    const lat = Number(locationLat);
    const lng = Number(locationLng);
    if (Number.isNaN(lat) || Number.isNaN(lng)) {
      setError("Provide valid location coordinates.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const found = await getNearbyFacilities({
        location_lat: lat,
        location_lng: lng,
        facility_type: emergencyOnly ? "emergency" : facilityType,
        specialization: specialization || undefined,
        radius_km: radiusKm,
      });
      setFacilities(found);
      setActiveTab("facilities");
      setNotice(found.length ? "Facilities loaded." : "No facilities found.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to fetch facilities.");
    } finally {
      setBusy(false);
    }
  }

  function removeComposerAttachment(): void {
    if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
    setImagePreviewUrl("");
    setImageFile(null);
  }

  function handleAttachmentSelection(file: File | null): void {
    if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
    setImageFile(file);
    setImagePreviewUrl(file ? URL.createObjectURL(file) : "");
  }

  async function submitAnalysis(params: {
    text: string;
    tags?: string[];
    file?: File | null;
    clearComposer?: boolean;
    modelProfile?: string;
  }): Promise<void> {
    const text = params.text.trim();
    if (!text) {
      setError("Please describe your symptoms first.");
      return;
    }

    const medicalHistory: Record<string, unknown> = {
      conditions: parseCommaList(profileConditionsText),
      allergies: parseCommaList(profileAllergiesText),
      medications: parseCommaList(profileMedicationsText),
      comorbidities: parseCommaList(profileComorbiditiesText),
      pregnancy_status: profilePregnancyStatus || "",
    };

    let sessionId = activeSessionId;
    const effectiveModelProfile = params.modelProfile || selectedModel;
    const tags = params.tags || [];
    const file = params.file || null;
    let uploadedFileUrl = "";

    setBusy(true);
    setIsTyping(true);
    setError("");

    try {
      if (!sessionId) {
        const created = await createSession();
        setSessions((prev) => [created, ...prev]);
        setActiveSessionId(created.id);
        sessionId = created.id;
      }

      const response = await analyzeInSession(sessionId, {
        symptomText: text,
        symptomTags: tags,
        imageFile: file,
        consentGiven,
        searchConsentGiven,
        modelProfile: effectiveModelProfile,
        facilityType,
        specialization,
        radiusKm,
        metadata: medicalHistory,
        location: locationLat && locationLng ? { lat: Number(locationLat), lng: Number(locationLng) } : null,
      });

      if (file) {
        uploadedFileUrl = URL.createObjectURL(file);
        setMessageAttachments((prev) => ({
          ...prev,
          [response.user_message.id]: {
            name: file.name,
            url: uploadedFileUrl,
            type: file.type,
            size: file.size,
          },
        }));
      }

      const payload = await getSessionMessages(sessionId);
      setMessages(payload.messages);
      setAnalysis(response.analysis);
      setFacilities(response.analysis.nearby_facilities || []);
      setStreamingMessageId(response.assistant_message.id);
      setActiveTab("chat");

      // REQ-8: Auto-switch to facilities tab when emergency is auto-triggered
      if (response.analysis.emergency_auto_triggered || response.analysis.needs_urgent_care) {
        setActiveTab("facilities");
      } else {
        setActiveTab("chat");
      }

      setNotice("Response ready.");

      if (params.clearComposer !== false) {
        setSymptomText("");
        setSymptomTagsText("");
        removeComposerAttachment();
      }
      await refreshSessions();
    } catch (err) {
      if (uploadedFileUrl) URL.revokeObjectURL(uploadedFileUrl);
      setError(err instanceof Error ? err.message : "Analysis request failed.");
    } finally {
      setBusy(false);
      setIsTyping(false);
    }
  }

  async function handleAnalyze(event: FormEvent<HTMLFormElement>, modelProfile: string): Promise<void> {
    event.preventDefault();
    const tags = symptomTagsText.split(",").map((item) => item.trim()).filter(Boolean);
    await submitAnalysis({ text: symptomText, tags, file: imageFile, clearComposer: true, modelProfile });
  }

  async function handleSaveProfile(): Promise<void> {
    if (!profile) return;
    setBusy(true);
    setError("");
    try {
      const medicalProfile = {
        conditions: parseCommaList(profileConditionsText),
        allergies: parseCommaList(profileAllergiesText),
        medications: parseCommaList(profileMedicationsText),
        comorbidities: parseCommaList(profileComorbiditiesText),
        pregnancy_status: profilePregnancyStatus || "",
        emergency_contact_email: profileEmergencyEmail || "",
      };
      const updated = await updateProfile({
        first_name: profileFirstName,
        last_name: profileLastName,
        email: profileEmail,
        phone_number: profilePhone,
        age: profileAge ? Number(profileAge) : null,
        gender: profileGender,
        emergency_contact_name: profileEmergencyName,
        emergency_contact_phone: profileEmergencyPhone,
        medical_profile: medicalProfile,
      });
      setProfile(updated);
      hydrateProfileForm(updated);
      setNotice("Profile updated.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Profile update failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleResendVerification(): Promise<void> {
    try {
      await resendVerificationEmail();
      setNotice("Verification email sent. Check your inbox.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to resend verification email.");
    }
  }

  async function handleExportProfile(): Promise<void> {
    try {
      const payload = await exportProfileJSON();
      downloadText("profile-export.json", JSON.stringify(payload, null, 2), "application/json");
      setNotice("Profile export downloaded.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to export profile.");
    }
  }

  async function handleExportChat(kind: "json" | "csv"): Promise<void> {
    try {
      if (kind === "json") {
        const payload = await exportChatHistoryJSON(activeSession?.id);
        downloadText("chat-history.json", JSON.stringify(payload, null, 2), "application/json");
      } else {
        const csv = await exportChatHistoryCSV(activeSession?.id);
        downloadText("chat-history.csv", csv, "text/csv");
      }
      setNotice(`Chat exported as ${kind.toUpperCase()}.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to export chat.");
    }
  }

  function handleCopyText(content: string, successMessage: string): void {
    navigator.clipboard
      .writeText(content)
      .then(() => setNotice(successMessage))
      .catch(() => setError("Unable to copy content."));
  }

  function handleVoiceInput(): void {
    const SpeechRecognition =
      (window as unknown as { SpeechRecognition?: typeof window.SpeechRecognition }).SpeechRecognition ||
      (window as unknown as { webkitSpeechRecognition?: typeof window.SpeechRecognition }).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setNotice("Voice input not supported in this browser.");
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.onresult = (event) => {
      const transcript = event.results?.[0]?.[0]?.transcript || "";
      setSymptomText((prev) => (prev ? `${prev} ${transcript}` : transcript));
    };
    recognition.start();
  }

  function handleCopyMessage(message: ChatMessage): void {
    handleCopyText(message.content || "", "Message copied.");
  }

  function handleDeleteMessage(messageId: number): void {
    setHiddenMessageIds((prev) => (prev.includes(messageId) ? prev : [...prev, messageId]));
    const attachment = messageAttachments[messageId];
    if (attachment) {
      URL.revokeObjectURL(attachment.url);
      setMessageAttachments((prev) => {
        const next = { ...prev };
        delete next[messageId];
        return next;
      });
    }
  }

  function handleFeedback(messageId: number, value: "up" | "down"): void {
    setFeedbackByMessage((prev) => ({ ...prev, [messageId]: value }));
    setNotice(value === "up" ? "Feedback saved: helpful." : "Feedback saved: needs improvement.");
  }

  async function handleRegenerate(messageId: number): Promise<void> {
    const index = messages.findIndex((item) => item.id === messageId);
    if (index <= 0) {
      setError("Unable to regenerate from this message.");
      return;
    }
    let sourceUserMessage: ChatMessage | null = null;
    for (let ptr = index - 1; ptr >= 0; ptr -= 1) {
      if (messages[ptr].role === "user") {
        sourceUserMessage = messages[ptr];
        break;
      }
    }
    if (!sourceUserMessage) {
      setError("No user message context found for regeneration.");
      return;
    }
    await submitAnalysis({
      text: sourceUserMessage.content,
      tags: [],
      file: null,
      clearComposer: false,
      modelProfile: selectedModel,
    });
  }

  function handleQuickPrompt(prompt: string): void {
    setActiveTab("chat");
    setSymptomText(prompt);
    requestAnimationFrame(() => composerRef.current?.focus());
  }

  function handleComposerKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>): void {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (busy || !symptomText.trim()) return;
      const tags = symptomTagsText.split(",").map((item) => item.trim()).filter(Boolean);
      void submitAnalysis({ text: symptomText, tags, file: imageFile, clearComposer: true, modelProfile: selectedModel });
    }
  }

  function openAttachment(messageId: number): void {
    const attachment = messageAttachments[messageId];
    if (!attachment) return;
    window.open(attachment.url, "_blank", "noopener,noreferrer");
  }

  function scrollChatToBottom(): void {
    const node = chatFeedRef.current;
    if (!node) return;
    node.scrollTo({ top: node.scrollHeight, behavior: "smooth" });
  }

  if (booting) {
    return <main className="page loading-page">Loading healthcare assistant...</main>;
  }

  if (!user || !profile) {
    return (
      <AuthScreen
        mode={mode}
        busy={busy}
        error={error}
        notice={notice}
        identifier={identifier}
        password={password}
        regEmail={regEmail}
        regPassword={regPassword}
        regFirstName={regFirstName}
        regLastName={regLastName}
        regPhone={regPhone}
        forgotEmail={forgotEmail}
        resetToken={resetToken}
        resetPassword={resetPassword}
        resetPasswordConfirm={resetPasswordConfirm}
        onModeChange={setMode}
        onIdentifierChange={setIdentifier}
        onPasswordChange={setPassword}
        onRegEmailChange={setRegEmail}
        onRegPasswordChange={setRegPassword}
        onRegFirstNameChange={setRegFirstName}
        onRegLastNameChange={setRegLastName}
        onRegPhoneChange={setRegPhone}
        onForgotEmailChange={setForgotEmail}
        onResetTokenChange={setResetToken}
        onResetPasswordChange={setResetPassword}
        onResetPasswordConfirmChange={setResetPasswordConfirm}
        onLogin={(event) => void handleLogin(event)}
        onRegister={(event) => void handleRegister(event)}
        onForgotPassword={(event) => void handleForgotPassword(event)}
        onResetPassword={(event) => void handleResetPassword(event)}
      />
    );
  }

  return (
    <main className="page app-page">
      <div
        className={mobileSidebarOpen ? "mobile-overlay show" : "mobile-overlay"}
        onClick={() => setMobileSidebarOpen(false)}
        aria-hidden="true"
      />

      <section className={sidebarCollapsed ? "chat-shell sidebar-collapsed" : "chat-shell"}>
        <AppSidebar
          sidebarCollapsed={sidebarCollapsed}
          mobileSidebarOpen={mobileSidebarOpen}
          theme={theme}
          selectedModel={selectedModel}
          busy={busy}
          sessions={sessions}
          activeSessionId={activeSessionId}
          onToggleSidebar={() => setSidebarCollapsed((prev) => !prev)}
          onToggleTheme={() => setTheme((prev) => (prev === "light" ? "dark" : "light"))}
          onCreateSession={() => void handleCreateSession()}
          onSelectedModelChange={setSelectedModel}
          onSelectSession={(sessionId) => void handleSelectSession(sessionId)}
          onOpenSettings={() => setSettingsOpen(true)}
          onExportProfile={() => void handleExportProfile()}
          onExportChatJson={() => void handleExportChat("json")}
          onExportChatCsv={() => void handleExportChat("csv")}
          onLogout={() => void handleLogout()}
        />

        <section className="main-panel">
          <MainHeader
            activeTab={activeTab}
            selectedModel={selectedModel}
            showAdminTab={Boolean(user?.is_staff)}
            onOpenMobileSidebar={() => setMobileSidebarOpen(true)}
            onTabChange={setActiveTab}
          />

          <section className="main-content">
            {activeTab === "chat" && (
              <ChatPane
                selectedModel={selectedModel}
                visibleMessages={visibleMessages}
                messageAttachments={messageAttachments}
                feedbackByMessage={feedbackByMessage}
                streamingMessageId={streamingMessageId}
                isTyping={isTyping}
                showScrollToBottom={showScrollToBottom}
                imagePreviewUrl={imagePreviewUrl}
                imageFile={imageFile}
                busy={busy}
                symptomText={symptomText}
                symptomTagsText={symptomTagsText}
                consentGiven={consentGiven}
                searchConsentGiven={searchConsentGiven}
                showComposerDetails={showComposerDetails}
                chatFeedRef={chatFeedRef}
                composerRef={composerRef}
                onAnalyzeSubmit={(event, modelProfile) => void handleAnalyze(event, modelProfile)}
                onQuickPrompt={handleQuickPrompt}
                onCopyMessage={handleCopyMessage}
                onRegenerate={(messageId) => void handleRegenerate(messageId)}
                onFeedback={handleFeedback}
                onDeleteMessage={handleDeleteMessage}
                onOpenAttachment={openAttachment}
                onCopyCode={(content) => handleCopyText(content, "Code copied.")}
                onVoiceInputNotice={handleVoiceInput}
                onScrollToBottom={scrollChatToBottom}
                onRemoveComposerAttachment={removeComposerAttachment}
                onSymptomTextChange={setSymptomText}
                onComposerKeyDown={handleComposerKeyDown}
                onAttachmentSelection={handleAttachmentSelection}
                onToggleComposerDetails={() => setShowComposerDetails((prev) => !prev)}
                onConsentChange={setConsentGiven}
                onSearchConsentChange={setSearchConsentGiven}
                onSymptomTagsTextChange={setSymptomTagsText}
                onOpenGuidance={() => setActiveTab("guidance")}
                onOpenFacilities={() => setActiveTab("facilities")}
              />
            )}

            {activeTab === "guidance" && (
              <GuidancePanel analysis={analysis} recentAssistantMessages={recentAssistantMessages} />
            )}

            {activeTab === "facilities" && (
              <FacilitiesPanel
                facilityType={facilityType}
                specialization={specialization}
                radiusKm={radiusKm}
                locationLat={locationLat}
                locationLng={locationLng}
                facilities={facilities}
                contacts={contacts}
                onFacilityTypeChange={setFacilityType}
                onSpecializationChange={setSpecialization}
                onRadiusKmChange={setRadiusKm}
                onLocationLatChange={setLocationLat}
                onLocationLngChange={setLocationLng}
                onDetectLocation={() => void handleDetectLocation()}
                onFacilitySearch={(emergencyOnly) => void handleFacilitySearch(emergencyOnly)}
              />
            )}

            {activeTab === "profile" && (
              <ProfilePanel
                busy={busy}
                firstName={profileFirstName}
                lastName={profileLastName}
                email={profileEmail}
                phone={profilePhone}
                age={profileAge}
                gender={profileGender}
                emergencyName={profileEmergencyName}
                emergencyPhone={profileEmergencyPhone}
                emergencyEmail={profileEmergencyEmail}
                conditionsText={profileConditionsText}
                allergiesText={profileAllergiesText}
                medicationsText={profileMedicationsText}
                comorbiditiesText={profileComorbiditiesText}
                pregnancyStatus={profilePregnancyStatus}
                emailVerified={profile?.email_verified}
                onFirstNameChange={setProfileFirstName}
                onLastNameChange={setProfileLastName}
                onEmailChange={setProfileEmail}
                onPhoneChange={setProfilePhone}
                onAgeChange={setProfileAge}
                onGenderChange={setProfileGender}
                onEmergencyNameChange={setProfileEmergencyName}
                onEmergencyPhoneChange={setProfileEmergencyPhone}
                onEmergencyEmailChange={setProfileEmergencyEmail}
                onConditionsChange={setProfileConditionsText}
                onAllergiesChange={setProfileAllergiesText}
                onMedicationsChange={setProfileMedicationsText}
                onComorbiditiesChange={setProfileComorbiditiesText}
                onPregnancyStatusChange={setProfilePregnancyStatus}
                onSave={() => void handleSaveProfile()}
                onResendVerification={() => void handleResendVerification()}
              />
            )}

            {activeTab === "admin" && user?.is_staff && <AdminDashboard />}
          </section>
        </section>
      </section>

      {(error || notice) && (
        <footer className="app-toast" role="status" aria-live="polite">
          {error && <p className="error">{error}</p>}
          {notice && <p className="status">{notice}</p>}
        </footer>
      )}

      {settingsOpen && (
        <SettingsModal
          theme={theme}
          selectedModel={selectedModel}
          showComposerDetails={showComposerDetails}
          onClose={() => setSettingsOpen(false)}
          onThemeChange={setTheme}
          onSelectedModelChange={setSelectedModel}
          onShowComposerDetailsChange={setShowComposerDetails}
        />
      )}
    </main>
  );
}
