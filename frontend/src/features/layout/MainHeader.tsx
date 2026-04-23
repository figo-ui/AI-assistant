interface MainHeaderProps {
  activeTab: "chat" | "guidance" | "facilities" | "profile" | "admin";
  selectedModel: string;
  showAdminTab?: boolean;
  onOpenMobileSidebar: () => void;
  onTabChange: (tab: "chat" | "guidance" | "facilities" | "profile" | "admin") => void;
}

export function MainHeader({
  activeTab,
  selectedModel,
  showAdminTab = false,
  onOpenMobileSidebar,
  onTabChange,
}: MainHeaderProps) {
  return (
    <header className="main-header sticky">
      <div className="header-left">
        <button type="button" className="icon-btn mobile-only" onClick={onOpenMobileSidebar}>
          Menu
        </button>
        <div className="assistant-meta">
          <div className="assistant-avatar">H</div>
          <div>
            <h2>Healthcare Assistant</h2>
            <p>
              <span className="status-dot" /> Online · {selectedModel}
            </p>
          </div>
        </div>
      </div>
      <nav className="tab-nav" aria-label="Main sections">
        <button type="button" className={activeTab === "chat" ? "active" : ""} onClick={() => onTabChange("chat")}>
          Chat
        </button>
        <button type="button" className={activeTab === "guidance" ? "active" : ""} onClick={() => onTabChange("guidance")}>
          Guidance
        </button>
        <button type="button" className={activeTab === "facilities" ? "active" : ""} onClick={() => onTabChange("facilities")}>
          Facilities
        </button>
        <button type="button" className={activeTab === "profile" ? "active" : ""} onClick={() => onTabChange("profile")}>
          Profile
        </button>
        {showAdminTab && (
          <button type="button" className={activeTab === "admin" ? "active" : ""} onClick={() => onTabChange("admin")}>
            Admin
          </button>
        )}
      </nav>
    </header>
  );
}
