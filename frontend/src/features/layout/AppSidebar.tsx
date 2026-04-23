import type { ChatSession } from "../../types/analysis";

interface AppSidebarProps {
  sidebarCollapsed: boolean;
  mobileSidebarOpen: boolean;
  theme: "light" | "dark";
  selectedModel: string;
  busy: boolean;
  sessions: ChatSession[];
  activeSessionId: number | null;
  onToggleSidebar: () => void;
  onToggleTheme: () => void;
  onCreateSession: () => void;
  onSelectedModelChange: (value: string) => void;
  onSelectSession: (sessionId: number) => void;
  onOpenSettings: () => void;
  onExportProfile: () => void;
  onExportChatJson: () => void;
  onExportChatCsv: () => void;
  onLogout: () => void;
}

export function AppSidebar({
  sidebarCollapsed,
  mobileSidebarOpen,
  theme,
  selectedModel,
  busy,
  sessions,
  activeSessionId,
  onToggleSidebar,
  onToggleTheme,
  onCreateSession,
  onSelectedModelChange,
  onSelectSession,
  onOpenSettings,
  onExportProfile,
  onExportChatJson,
  onExportChatCsv,
  onLogout,
}: AppSidebarProps) {
  return (
    <aside className={mobileSidebarOpen ? "sidebar open-mobile" : "sidebar"}>
      <div className="sidebar-top">
        <div className="brand-row">
          <div className="brand-avatar">AI</div>
          <div className="brand-text">
            <h1>Healthcare Assistant</h1>
            <p>Always-on clinical guidance</p>
          </div>
        </div>

        <div className="sidebar-control-row">
          <button type="button" className="icon-btn" onClick={onToggleSidebar}>
            {sidebarCollapsed ? "Expand" : "Collapse"}
          </button>
          <button type="button" className="icon-btn" onClick={onToggleTheme}>
            {theme === "light" ? "Dark" : "Light"}
          </button>
        </div>

        <button type="button" className="new-chat-btn" onClick={onCreateSession} disabled={busy}>
          + New Chat
        </button>

        <label className="model-picker">
          Model
          <select value={selectedModel} onChange={(event) => onSelectedModelChange(event.target.value)}>
            <option value="Clinical Balanced">Clinical Balanced</option>
            <option value="Clinical Fast">Clinical Fast</option>
            <option value="Clinical Thorough">Clinical Thorough</option>
          </select>
        </label>
      </div>

      <div className="sidebar-history">
        <h3>Conversations</h3>
        <div className="session-list">
          {sessions.map((entry) => (
            <button
              key={entry.id}
              type="button"
              className={entry.id === activeSessionId ? "session-item active" : "session-item"}
              onClick={() => onSelectSession(entry.id)}
            >
              <strong>{entry.title}</strong>
              <small>{entry.last_message || "No messages yet"}</small>
            </button>
          ))}
        </div>
      </div>

      <div className="sidebar-bottom">
        <button type="button" className="ghost" onClick={onOpenSettings}>
          Settings
        </button>
        <button type="button" className="ghost" onClick={onExportProfile}>
          Export profile
        </button>
        <button type="button" className="ghost" onClick={onExportChatJson}>
          Export JSON
        </button>
        <button type="button" className="ghost" onClick={onExportChatCsv}>
          Export CSV
        </button>
        <button type="button" className="danger" onClick={onLogout}>
          Logout
        </button>
      </div>
    </aside>
  );
}
