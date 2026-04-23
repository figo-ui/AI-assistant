import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { AdminAnalytics, AdminFacility, AuditLogEntry, User } from "../../types/analysis";
import {
  createAdminFacility,
  deleteAdminFacility,
  getAdminAnalytics,
  getAdminAuditLog,
  getAdminDialogueTemplates,
  getAdminModelMetrics,
  listAdminFacilities,
  listAdminUsers,
  triggerAdminRetrain,
  updateAdminDialogueTemplate,
  updateAdminFacility,
  updateAdminUser,
  type FacilityPayload,
} from "../../utils/apiClient";

type AdminTab = "analytics" | "users" | "facilities" | "audit" | "metrics" | "templates";

const DEFAULT_FACILITY: FacilityPayload = {
  name: "",
  facility_type: "hospital",
  specialization: "",
  address: "",
  phone_number: "",
  latitude: 0,
  longitude: 0,
  is_emergency: false,
};

function toFacilityPayload(item: AdminFacility): FacilityPayload {
  return {
    name: item.name || "",
    facility_type: item.facility_type || "hospital",
    specialization: item.specialization || "",
    address: item.address || "",
    phone_number: item.phone_number || "",
    latitude: Number(item.latitude ?? 0),
    longitude: Number(item.longitude ?? 0),
    is_emergency: Boolean(item.is_emergency),
  };
}

export function AdminDashboard() {
  const [activeTab, setActiveTab] = useState<AdminTab>("analytics");
  const [analytics, setAnalytics] = useState<AdminAnalytics | null>(null);
  const [facilities, setFacilities] = useState<AdminFacility[]>([]);
  const [users, setUsers] = useState<User[]>([]);
  const [auditLog, setAuditLog] = useState<AuditLogEntry[]>([]);
  const [auditTotal, setAuditTotal] = useState(0);
  const [auditPage, setAuditPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [newFacility, setNewFacility] = useState<FacilityPayload>({ ...DEFAULT_FACILITY });
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editingDraft, setEditingDraft] = useState<FacilityPayload | null>(null);
  const [retrainBusy, setRetrainBusy] = useState(false);
  const [modelMetrics, setModelMetrics] = useState<Record<string, unknown> | null>(null);
  const [dialogueTemplates, setDialogueTemplates] = useState<Record<string, string[]>>({});
  const [editingIntent, setEditingIntent] = useState<string | null>(null);
  const [editingTemplateText, setEditingTemplateText] = useState("");

  const riskData = useMemo(() => {
    const breakdown = analytics?.risk_breakdown || [];
    const lookup = new Map(breakdown.map((item) => [String(item.risk__risk_level), Number(item.count || 0)]));
    return ["Low", "Medium", "High"].map((level) => ({
      level,
      count: lookup.get(level) ?? 0,
    }));
  }, [analytics]);

  async function loadAll(): Promise<void> {
    setLoading(true);
    setError("");
    try {
      const [analyticsPayload, facilitiesPayload, usersPayload] = await Promise.all([
        getAdminAnalytics(),
        listAdminFacilities(),
        listAdminUsers(),
      ]);
      setAnalytics(analyticsPayload);
      setFacilities(facilitiesPayload);
      setUsers(usersPayload.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load admin data.");
    } finally {
      setLoading(false);
    }
  }

  async function loadAuditLog(page = 1): Promise<void> {
    try {
      const payload = await getAdminAuditLog({ page, page_size: 20 });
      setAuditLog(payload.results);
      setAuditTotal(payload.total);
      setAuditPage(page);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load audit log.");
    }
  }

  useEffect(() => {
    void loadAll();
  }, []);

  useEffect(() => {
    if (activeTab === "audit") void loadAuditLog(1);
    if (activeTab === "metrics") void loadModelMetrics();
    if (activeTab === "templates") void loadDialogueTemplates();
  }, [activeTab]);

  async function loadModelMetrics(): Promise<void> {
    try {
      const data = await getAdminModelMetrics();
      setModelMetrics(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load model metrics.");
    }
  }

  async function loadDialogueTemplates(): Promise<void> {
    try {
      const data = await getAdminDialogueTemplates();
      setDialogueTemplates(data.templates || {});
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load dialogue templates.");
    }
  }

  async function handleSaveTemplate(intent: string): Promise<void> {
    const lines = editingTemplateText.split("\n").map((l) => l.trim()).filter(Boolean);
    try {
      await updateAdminDialogueTemplate(intent, lines);
      setDialogueTemplates((prev) => ({ ...prev, [intent]: lines }));
      setEditingIntent(null);
      setEditingTemplateText("");
      setNotice(`Template for "${intent}" updated.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to save template.");
    }
  }

  async function handleToggleUserActive(user: User): Promise<void> {
    try {
      const updated = await updateAdminUser(user.id, { is_active: !user.is_active });
      setUsers((prev) => prev.map((u) => (u.id === updated.id ? updated : u)));
      setNotice(`User ${updated.username} ${updated.is_active ? "activated" : "deactivated"}.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to update user.");
    }
  }

  async function handleToggleUserStaff(user: User): Promise<void> {
    try {
      const updated = await updateAdminUser(user.id, { is_staff: !user.is_staff });
      setUsers((prev) => prev.map((u) => (u.id === updated.id ? updated : u)));
      setNotice(`User ${updated.username} staff status updated.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to update user.");
    }
  }

  async function handleRetrain(): Promise<void> {
    setRetrainBusy(true);
    setError("");
    try {
      const result = await triggerAdminRetrain();
      setNotice(result.message || "Retrain request submitted.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Retrain request failed.");
    } finally {
      setRetrainBusy(false);
    }
  }

  async function handleCreateFacility(): Promise<void> {
    if (!newFacility.name.trim()) {
      setError("Facility name is required.");
      return;
    }
    try {
      const created = await createAdminFacility(newFacility);
      setFacilities((prev) => [...prev, created]);
      setNewFacility({ ...DEFAULT_FACILITY });
      setNotice("Facility created.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to create facility.");
    }
  }

  async function handleSaveFacility(facilityId: number): Promise<void> {
    if (!editingDraft) return;
    try {
      const updated = await updateAdminFacility(facilityId, editingDraft);
      setFacilities((prev) => prev.map((f) => (f.id === updated.id ? updated : f)));
      setEditingId(null);
      setEditingDraft(null);
      setNotice("Facility updated.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to update facility.");
    }
  }

  async function handleDeleteFacility(facilityId: number): Promise<void> {
    if (!window.confirm("Delete this facility?")) return;
    try {
      await deleteAdminFacility(facilityId);
      setFacilities((prev) => prev.filter((f) => f.id !== facilityId));
      setNotice("Facility deleted.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to delete facility.");
    }
  }

  if (loading) return <section className="content-card"><p>Loading admin data...</p></section>;

  return (
    <section className="content-card">
      <h2>Admin Dashboard</h2>
      {error && <p className="error">{error}</p>}
      {notice && <p className="status">{notice}</p>}

      <nav className="tab-nav" style={{ marginBottom: "1rem" }}>
        {(["analytics", "users", "facilities", "audit", "metrics", "templates"] as AdminTab[]).map((tab) => (
          <button
            key={tab}
            type="button"
            className={activeTab === tab ? "active" : ""}
            onClick={() => setActiveTab(tab)}
          >
            {tab === "metrics" ? "Model Metrics" : tab === "templates" ? "Dialogue" : tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </nav>

      {/* ── Analytics Tab ── */}
      {activeTab === "analytics" && analytics && (
        <>
          <div className="row-3" style={{ gap: "1rem", marginBottom: "1rem" }}>
            {[
              ["Total Users", analytics.users_total],
              ["Active Users", analytics.users_active],
              ["Chat Sessions", analytics.chat_sessions_total],
              ["Cases Total", analytics.cases_total],
              ["Cases Today", analytics.cases_today ?? 0],
              ["Facilities", analytics.facilities_total],
            ].map(([label, value]) => (
              <div key={String(label)} className="facility-item" style={{ textAlign: "center" }}>
                <strong style={{ fontSize: "1.4rem" }}>{value}</strong>
                <p className="muted">{label}</p>
              </div>
            ))}
          </div>

          <h3>Risk Distribution</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={riskData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="level" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="count" fill="#4f8ef7" />
            </BarChart>
          </ResponsiveContainer>

          <div style={{ marginTop: "1.5rem" }}>
            <h3>Model Retraining</h3>
            <p className="muted">
              Triggers a retraining request. Run{" "}
              <code>backend/scripts/preprocess_and_train.py</code> on the server to execute.
            </p>
            <button type="button" onClick={() => void handleRetrain()} disabled={retrainBusy}>
              {retrainBusy ? "Requesting..." : "Request Model Retrain"}
            </button>
          </div>
        </>
      )}

      {/* ── Users Tab ── */}
      {activeTab === "users" && (
        <>
          <h3>Users ({users.length})</h3>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
              <thead>
                <tr>
                  {["ID", "Username", "Email", "Active", "Staff", "Actions"].map((h) => (
                    <th key={h} style={{ textAlign: "left", padding: "6px 8px", borderBottom: "1px solid var(--border)" }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {users.map((u) => (
                  <tr key={u.id}>
                    <td style={{ padding: "6px 8px" }}>{u.id}</td>
                    <td style={{ padding: "6px 8px" }}>{u.username}</td>
                    <td style={{ padding: "6px 8px" }}>{u.email}</td>
                    <td style={{ padding: "6px 8px" }}>
                      <span style={{ color: u.is_active ? "green" : "red" }}>
                        {u.is_active ? "Yes" : "No"}
                      </span>
                    </td>
                    <td style={{ padding: "6px 8px" }}>
                      <span style={{ color: u.is_staff ? "var(--accent)" : "inherit" }}>
                        {u.is_staff ? "Yes" : "No"}
                      </span>
                    </td>
                    <td style={{ padding: "6px 8px", display: "flex", gap: "6px" }}>
                      <button
                        type="button"
                        className="ghost"
                        style={{ fontSize: "0.8rem", padding: "2px 8px" }}
                        onClick={() => void handleToggleUserActive(u)}
                      >
                        {u.is_active ? "Deactivate" : "Activate"}
                      </button>
                      <button
                        type="button"
                        className="ghost"
                        style={{ fontSize: "0.8rem", padding: "2px 8px" }}
                        onClick={() => void handleToggleUserStaff(u)}
                      >
                        {u.is_staff ? "Remove Staff" : "Make Staff"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* ── Facilities Tab ── */}
      {activeTab === "facilities" && (
        <>
          <h3>Add Facility</h3>
          <div className="row-3" style={{ gap: "0.5rem", flexWrap: "wrap" }}>
            {(["name", "address", "phone_number", "specialization"] as const).map((field) => (
              <label key={field}>
                {field.replace(/_/g, " ")}
                <input
                  value={String(newFacility[field] ?? "")}
                  onChange={(e) => setNewFacility((prev) => ({ ...prev, [field]: e.target.value }))}
                />
              </label>
            ))}
            <label>
              Type
              <select
                value={newFacility.facility_type}
                onChange={(e) => setNewFacility((prev) => ({ ...prev, facility_type: e.target.value }))}
              >
                {["hospital", "clinic", "pharmacy", "emergency"].map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </label>
            <label>
              Latitude
              <input
                type="number"
                value={newFacility.latitude}
                onChange={(e) => setNewFacility((prev) => ({ ...prev, latitude: Number(e.target.value) }))}
              />
            </label>
            <label>
              Longitude
              <input
                type="number"
                value={newFacility.longitude}
                onChange={(e) => setNewFacility((prev) => ({ ...prev, longitude: Number(e.target.value) }))}
              />
            </label>
            <label>
              Emergency
              <input
                type="checkbox"
                checked={newFacility.is_emergency}
                onChange={(e) => setNewFacility((prev) => ({ ...prev, is_emergency: e.target.checked }))}
              />
            </label>
          </div>
          <button type="button" onClick={() => void handleCreateFacility()} style={{ marginTop: "0.5rem" }}>
            Add Facility
          </button>

          <h3 style={{ marginTop: "1.5rem" }}>Existing Facilities ({facilities.length})</h3>
          <div className="facility-list">
            {facilities.map((f) => (
              <article key={f.id} className="facility-item">
                {editingId === f.id && editingDraft ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                    {(["name", "address", "phone_number"] as const).map((field) => (
                      <input
                        key={field}
                        placeholder={field}
                        value={String(editingDraft[field] ?? "")}
                        onChange={(e) => setEditingDraft((prev) => prev ? { ...prev, [field]: e.target.value } : prev)}
                      />
                    ))}
                    <label>
                      Emergency
                      <input
                        type="checkbox"
                        checked={editingDraft.is_emergency}
                        onChange={(e) => setEditingDraft((prev) => prev ? { ...prev, is_emergency: e.target.checked } : prev)}
                      />
                    </label>
                    <div style={{ display: "flex", gap: "6px" }}>
                      <button type="button" onClick={() => void handleSaveFacility(f.id)}>Save</button>
                      <button type="button" className="ghost" onClick={() => { setEditingId(null); setEditingDraft(null); }}>Cancel</button>
                    </div>
                  </div>
                ) : (
                  <>
                    <strong>{f.name}</strong>
                    <p>{f.facility_type}{f.is_emergency ? " · Emergency" : ""}</p>
                    <p className="muted">{f.address}</p>
                    <div style={{ display: "flex", gap: "6px", marginTop: "4px" }}>
                      <button
                        type="button"
                        className="ghost"
                        style={{ fontSize: "0.8rem" }}
                        onClick={() => { setEditingId(f.id); setEditingDraft(toFacilityPayload(f)); }}
                      >
                        Edit
                      </button>
                      <button
                        type="button"
                        className="danger"
                        style={{ fontSize: "0.8rem" }}
                        onClick={() => void handleDeleteFacility(f.id)}
                      >
                        Delete
                      </button>
                    </div>
                  </>
                )}
              </article>
            ))}
          </div>
        </>
      )}

      {/* ── Audit Log Tab ── */}
      {activeTab === "audit" && (
        <>
          <h3>Audit Log ({auditTotal} entries)</h3>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
              <thead>
                <tr>
                  {["Time", "Actor", "Action", "Target", "ID"].map((h) => (
                    <th key={h} style={{ textAlign: "left", padding: "6px 8px", borderBottom: "1px solid var(--border)" }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {auditLog.map((entry) => (
                  <tr key={entry.id}>
                    <td style={{ padding: "6px 8px", whiteSpace: "nowrap" }}>
                      {new Date(entry.created_at).toLocaleString()}
                    </td>
                    <td style={{ padding: "6px 8px" }}>{entry.actor ?? "system"}</td>
                    <td style={{ padding: "6px 8px" }}>{entry.action}</td>
                    <td style={{ padding: "6px 8px" }}>{entry.target_type}</td>
                    <td style={{ padding: "6px 8px" }}>{entry.target_id}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ display: "flex", gap: "8px", marginTop: "0.75rem" }}>
            <button
              type="button"
              className="ghost"
              disabled={auditPage <= 1}
              onClick={() => void loadAuditLog(auditPage - 1)}
            >
              Previous
            </button>
            <span style={{ lineHeight: "2rem" }}>Page {auditPage}</span>
            <button
              type="button"
              className="ghost"
              disabled={auditPage * 20 >= auditTotal}
              onClick={() => void loadAuditLog(auditPage + 1)}
            >
              Next
            </button>
          </div>
        </>
      )}

      {/* ── Model Metrics Tab ── */}
      {activeTab === "metrics" && (
        <>
          <h3>Model Performance Metrics</h3>
          {modelMetrics ? (
            <>
              {(["text_model", "image_model", "dialogue_model"] as const).map((key) => {
                const section = modelMetrics[key] as Record<string, unknown> | undefined;
                if (!section) return null;
                return (
                  <div key={key} style={{ marginBottom: "1.5rem" }}>
                    <h4 style={{ textTransform: "capitalize", marginBottom: "6px" }}>
                      {key.replace(/_/g, " ")}
                    </h4>
                    {Object.entries(section).map(([subKey, subVal]) => (
                      <div key={subKey} style={{ marginBottom: "8px" }}>
                        <strong style={{ fontSize: "0.85rem", textTransform: "capitalize" }}>
                          {subKey.replace(/_/g, " ")}
                        </strong>
                        <pre
                          style={{
                            background: "var(--surface-2, #f4f4f5)",
                            borderRadius: "4px",
                            padding: "8px",
                            fontSize: "0.78rem",
                            overflowX: "auto",
                            marginTop: "4px",
                          }}
                        >
                          {JSON.stringify(subVal, null, 2)}
                        </pre>
                      </div>
                    ))}
                  </div>
                );
              })}
            </>
          ) : (
            <p className="muted">Loading metrics...</p>
          )}
        </>
      )}

      {/* ── Dialogue Templates Tab ── */}
      {activeTab === "templates" && (
        <>
          <h3>Dialogue Response Templates</h3>
          <p className="muted" style={{ fontSize: "0.85rem", marginBottom: "1rem" }}>
            Edit the response templates used by the dialogue intent model. One response per line.
          </p>
          {Object.keys(dialogueTemplates).length === 0 && (
            <p className="muted">No templates loaded.</p>
          )}
          {Object.entries(dialogueTemplates).map(([intent, templates]) => (
            <div key={intent} style={{ marginBottom: "1rem", borderBottom: "1px solid var(--border)", paddingBottom: "1rem" }}>
              <strong style={{ fontSize: "0.9rem" }}>{intent}</strong>
              {editingIntent === intent ? (
                <div style={{ marginTop: "6px" }}>
                  <textarea
                    rows={Math.max(3, templates.length + 1)}
                    value={editingTemplateText}
                    onChange={(e) => setEditingTemplateText(e.target.value)}
                    style={{ width: "100%", fontFamily: "monospace", fontSize: "0.82rem", padding: "6px" }}
                  />
                  <div style={{ display: "flex", gap: "8px", marginTop: "6px" }}>
                    <button type="button" onClick={() => void handleSaveTemplate(intent)}>Save</button>
                    <button type="button" className="ghost" onClick={() => { setEditingIntent(null); setEditingTemplateText(""); }}>Cancel</button>
                  </div>
                </div>
              ) : (
                <div style={{ marginTop: "4px" }}>
                  <ul style={{ fontSize: "0.82rem", margin: "4px 0 6px 16px" }}>
                    {templates.map((t, i) => <li key={i}>{t}</li>)}
                  </ul>
                  <button
                    type="button"
                    className="ghost"
                    style={{ fontSize: "0.8rem" }}
                    onClick={() => {
                      setEditingIntent(intent);
                      setEditingTemplateText(templates.join("\n"));
                    }}
                  >
                    Edit
                  </button>
                </div>
              )}
            </div>
          ))}
        </>
      )}
    </section>
  );
}
