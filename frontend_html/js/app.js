/**
 * app.js — Main application controller.
 * Manages global state, routing between tabs, and coordinates all modules.
 */

const App = {

  /* ── State ─────────────────────────────────────────────── */
  state: {
    user:          null,
    profile:       null,
    sessions:      [],
    activeSession: null,
    messages:      [],
    analysis:      null,
    facilities:    [],
    contacts:      [],
    selectedModel: 'Clinical Balanced',
    activeTab:     'chat',
    quickPrompts:  [],   // loaded from server, not hardcoded
    busy:          false,
  },

  /* ── Boot ──────────────────────────────────────────────── */
  async init() {
    applyStoredTheme();
    document.addEventListener('keydown', e => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        App.switchTab('chat');
        setTimeout(() => document.getElementById('composer-input')?.focus(), 50);
      }
      if (e.key === 'Escape') {
        closeMobileSidebar();
        closeSettings();
      }
    });
    document.getElementById('mobile-overlay').addEventListener('click', closeMobileSidebar);

    // Try to restore session from stored tokens
    const tokens = getTokens();
    if (tokens?.access) {
      try {
        const profile = await authApi.getProfile();
        await App.onLoginSuccess(profile.user, profile);
        return;
      } catch { clearTokens(); }
    }
    App.showAuth();
  },

  /* ── Auth flow ─────────────────────────────────────────── */
  showAuth() {
    document.getElementById('app-screen').style.display = 'none';
    document.getElementById('auth-screen').style.display = 'grid';
    renderAuthPage();
  },

  async onLoginSuccess(user, profile) {
    App.state.user    = user;
    App.state.profile = profile;

    document.getElementById('auth-screen').style.display = 'none';
    document.getElementById('app-screen').style.display  = 'block';

    renderSidebar();
    applyStoredTheme();
    renderHeader('chat', App.state.selectedModel, user.is_staff);

    // Load emergency contacts
    facilitiesApi.emergency().then(d => {
      App.state.contacts = d.contacts || [];
    }).catch(() => {});

    // Load sessions
    await App.loadSessions();
    App.switchTab('chat');
  },

  async logout() {
    try {
      const t = getTokens();
      if (t?.refresh) await authApi.logout(t.refresh);
    } catch {}
    clearTokens();
    App.state = { ...App.state, user: null, profile: null, sessions: [], activeSession: null, messages: [], analysis: null };
    App.showAuth();
  },

  /* ── Quick prompts — loaded from server ────────────────── */
  async loadQuickPrompts() {
    try {
      // Public endpoint — no auth required, no hardcoded strings
      const data = await api.get('/quick-prompts/', false);
      App.state.quickPrompts = data.prompts || [];
    } catch {
      App.state.quickPrompts = [];
    }
  },

  /* ── Sessions ──────────────────────────────────────────── */
  async loadSessions() {
    try {
      const sessions = await chatApi.listSessions();
      App.state.sessions = sessions;
      renderSessionList(sessions, App.state.activeSession?.id || null);
      if (sessions.length && !App.state.activeSession) {
        await App.selectSession(sessions[0].id);
      }
    } catch (err) {
      showError(err.message || 'Could not load sessions.');
    }
  },

  async createSession() {
    try {
      const session = await chatApi.createSession();
      App.state.sessions.unshift(session);
      App.state.activeSession = session;
      App.state.messages = [];
      App.state.analysis = null;
      renderSessionList(App.state.sessions, session.id);
      App.switchTab('chat');
      renderMessages([]);
      showSuccess('New conversation started.');
    } catch (err) {
      showError(err.message || 'Could not create session.');
    }
  },

  async selectSession(id) {
    try {
      const data = await chatApi.getMessages(id);
      App.state.activeSession = data.session;
      App.state.messages = data.messages || [];
      // Find latest analysis
      const assistantMsgs = App.state.messages.filter(m => m.role === 'assistant').reverse();
      for (const m of assistantMsgs) {
        const a = parseAnalysis(m);
        if (a) { App.state.analysis = a; break; }
      }
      renderSessionList(App.state.sessions, id);
      if (App.state.activeTab === 'chat') renderMessages(App.state.messages);
      closeMobileSidebar();
    } catch (err) {
      showError(err.message || 'Could not open session.');
    }
  },

  /* ── Analysis ──────────────────────────────────────────── */
  async submitAnalysis(e) {
    if (e) e.preventDefault();
    const { symptomText, tags, consentGiven, searchConsent, imageFile } = getComposerValues();
    if (!symptomText) { showError('Please describe your symptoms first.'); return; }

    App.state.busy = true;
    setComposerBusy(true);
    showTyping();

    try {
      // Create session if none exists
      if (!App.state.activeSession) await App.createSession();
      const sessionId = App.state.activeSession.id;

      const med = App.state.profile?.medical_profile || App.state.profile?.medical_history || {};
      const payload = {
        symptomText,
        tags,
        consentGiven,
        searchConsentGiven: searchConsent,
        imageFile,
        modelProfile: App.state.selectedModel,
        metadata: {
          conditions:    med.conditions || [],
          allergies:     med.allergies  || [],
          medications:   med.medications || [],
          comorbidities: med.comorbidities || [],
          pregnancy_status: med.pregnancy_status || '',
        },
      };

      const res = await chatApi.analyze(sessionId, payload);
      App.state.analysis = res.analysis;

      // Refresh messages from server for consistency
      const msgData = await chatApi.getMessages(sessionId);
      App.state.messages = msgData.messages || [];
      renderMessages(App.state.messages);

      // Auto-switch to facilities if emergency
      if (res.analysis?.emergency_auto_triggered || res.analysis?.needs_urgent_care) {
        App.state.facilities = res.analysis.nearby_facilities || [];
        App.switchTab('facilities');
      }

      clearComposer();
      await App.loadSessions();
    } catch (err) {
      showError(err.message || 'Analysis failed. Please try again.');
    } finally {
      App.state.busy = false;
      setComposerBusy(false);
      hideTyping();
    }
  },

  async regenerate(messageId) {
    const msgs = App.state.messages;
    const idx  = msgs.findIndex(m => m.id === messageId);
    if (idx <= 0) return;
    let userMsg = null;
    for (let i = idx - 1; i >= 0; i--) {
      if (msgs[i].role === 'user') { userMsg = msgs[i]; break; }
    }
    if (!userMsg) return;
    // Inject text into composer and submit
    const ta = document.getElementById('composer-input');
    if (ta) { ta.value = userMsg.content; autoResize(ta); }
    await App.submitAnalysis(null);
  },

  feedback(messageId, value) {
    messageFeedback[messageId] = value;
    // Re-render just the action buttons for that message
    const row = document.getElementById(`msg-${messageId}`);
    if (row) {
      const upBtn   = row.querySelector('button[onclick*="up"]');
      const downBtn = row.querySelector('button[onclick*="down"]');
      if (upBtn)   upBtn.classList.toggle('active', value === 'up');
      if (downBtn) downBtn.classList.toggle('active', value === 'down');
    }
    showSuccess(value === 'up' ? 'Feedback saved: helpful.' : 'Feedback saved: needs improvement.');
  },

  deleteMessage(messageId) {
    hiddenMessages.add(messageId);
    const el = document.getElementById(`msg-${messageId}`);
    if (el) el.remove();
  },

  useQuickPrompt(text) {
    // Focus the composer so the user can type their own message
    const ta = document.getElementById('composer-input');
    if (ta) { ta.focus(); }
  },

  /* ── Tab routing ───────────────────────────────────────── */
  switchTab(tab) {
    App.state.activeTab = tab;
    renderHeader(tab, App.state.selectedModel, App.state.user?.is_staff);

    const content = document.getElementById('main-content');
    switch (tab) {
      case 'chat':
        renderChatPage();
        renderMessages(App.state.messages);
        break;
      case 'guidance':
        renderGuidancePage(App.state.analysis);
        break;
      case 'facilities':
        renderFacilitiesPage(App.state.facilities, App.state.contacts);
        break;
      case 'profile':
        renderProfilePage(App.state.profile);
        break;
      case 'admin':
        renderAdminShell();
        break;
    }
  },

  /* ── Model ─────────────────────────────────────────────── */
  setModel(model) {
    App.state.selectedModel = model;
    updateHeaderModel(model);
    const sel = document.getElementById('model-select');
    if (sel) sel.value = model;
  },

  /* ── Exports ───────────────────────────────────────────── */
  async exportProfile() {
    try {
      const data = await chatApi.exportProfile();
      downloadText('profile-export.json', JSON.stringify(data, null, 2), 'application/json');
      showSuccess('Profile exported.');
    } catch (err) { showError(err.message || 'Export failed.'); }
  },

  async exportChat(format) {
    const sessionId = App.state.activeSession?.id;
    try {
      if (format === 'json') {
        const data = await chatApi.exportJson(sessionId);
        downloadText('chat-history.json', JSON.stringify(data, null, 2), 'application/json');
      } else {
        const csv = await chatApi.exportCsv(sessionId);
        downloadText('chat-history.csv', csv, 'text/csv');
      }
      showSuccess(`Chat exported as ${format.toUpperCase()}.`);
    } catch (err) { showError(err.message || 'Export failed.'); }
  },
};

/* ── Start ─────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => App.init());
