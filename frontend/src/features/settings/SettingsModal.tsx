interface SettingsModalProps {
  theme: "light" | "dark";
  selectedModel: string;
  showComposerDetails: boolean;
  onClose: () => void;
  onThemeChange: (value: "light" | "dark") => void;
  onSelectedModelChange: (value: string) => void;
  onShowComposerDetailsChange: (value: boolean) => void;
}

export function SettingsModal({
  theme,
  selectedModel,
  showComposerDetails,
  onClose,
  onThemeChange,
  onSelectedModelChange,
  onShowComposerDetailsChange,
}: SettingsModalProps) {
  return (
    <div className="settings-modal-backdrop" onClick={onClose}>
      <section className="settings-modal" onClick={(event) => event.stopPropagation()}>
        <h3>Settings</h3>
        <label>
          Theme
          <select value={theme} onChange={(event) => onThemeChange(event.target.value as "light" | "dark")}>
            <option value="light">Light</option>
            <option value="dark">Dark</option>
          </select>
        </label>
        <label>
          Model profile
          <select value={selectedModel} onChange={(event) => onSelectedModelChange(event.target.value)}>
            <option value="Clinical Balanced">Clinical Balanced</option>
            <option value="Clinical Fast">Clinical Fast</option>
            <option value="Clinical Thorough">Clinical Thorough</option>
          </select>
        </label>
        <label className="checkbox-line">
          <input
            type="checkbox"
            checked={showComposerDetails}
            onChange={(event) => onShowComposerDetailsChange(event.target.checked)}
          />
          Show advanced composer options by default
        </label>
        <div className="settings-actions">
          <button type="button" onClick={onClose}>
            Done
          </button>
        </div>
      </section>
    </div>
  );
}
