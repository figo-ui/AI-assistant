import type { AnalysisResponse, ChatMessage } from "../../types/analysis";
import { displayConditions } from "../chat/chatUtils";

interface GuidancePanelProps {
  analysis: AnalysisResponse | null;
  recentAssistantMessages: ChatMessage[];
}

export function GuidancePanel({ analysis, recentAssistantMessages }: GuidancePanelProps) {
  const conditions = analysis ? displayConditions(analysis, 3) : [];

  return (
    <section className="content-card">
      <h2>Guidance Summary</h2>
      {analysis ? (
        <>
          <p>
            Risk level: <strong>{analysis.risk_level}</strong> ({analysis.risk_score.toFixed(2)})
          </p>
          <h3>Likely health patterns</h3>
          {conditions.length > 0 ? (
            <ul>
              {conditions.map((item) => (
                <li key={item.condition}>
                  {item.condition} ({(item.probability * 100).toFixed(1)}%)
                </li>
              ))}
            </ul>
          ) : (
            <p>No specific named condition identified from current data.</p>
          )}
          <h3>Recommended next step</h3>
          <p>{analysis.recommendation_text}</p>
          <h3>Prevention advice</h3>
          <ul>{analysis.prevention_advice.map((item) => <li key={item}>{item}</li>)}</ul>
          {analysis.risk_factors.length > 0 && (
            <>
              <h3>Why this risk level</h3>
              <ul>{analysis.risk_factors.slice(0, 4).map((item) => <li key={item}>{item}</li>)}</ul>
            </>
          )}
          <p className="muted">{analysis.disclaimer_text}</p>
        </>
      ) : (
        <p className="muted">No analysis yet. Start a chat message to generate guidance.</p>
      )}

      <h3>Recent Assistant Updates</h3>
      <div className="facility-list">
        {recentAssistantMessages.length > 0 ? (
          recentAssistantMessages.map((entry) => (
            <article key={entry.id} className="facility-item">
              <strong>{new Date(entry.created_at).toLocaleString()}</strong>
              <p>{entry.content}</p>
            </article>
          ))
        ) : (
          <p className="muted">No assistant updates yet.</p>
        )}
      </div>
    </section>
  );
}
