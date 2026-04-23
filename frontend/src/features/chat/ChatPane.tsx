import { Fragment, type FormEvent, type ReactNode, type RefObject } from "react";

import type { AnalysisResponse, ChatMessage } from "../../types/analysis";
import {
  QUICK_PROMPTS,
  dayKey,
  displayConditions,
  formatBytes,
  formatDayLabel,
  parseMessageAnalysis,
  segmentMessageContent,
  shouldShowTimestamp,
  type LocalAttachment,
} from "./chatUtils";

const INLINE_CODE_RE = /`([^`]+)`/g;

interface ChatPaneProps {
  selectedModel: string;
  visibleMessages: ChatMessage[];
  messageAttachments: Record<number, LocalAttachment>;
  feedbackByMessage: Record<number, "up" | "down">;
  streamingMessageId: number | null;
  isTyping: boolean;
  showScrollToBottom: boolean;
  imagePreviewUrl: string;
  imageFile: File | null;
  busy: boolean;
  symptomText: string;
  symptomTagsText: string;
  consentGiven: boolean;
  searchConsentGiven: boolean;
  showComposerDetails: boolean;
  chatFeedRef: RefObject<HTMLDivElement | null>;
  composerRef: RefObject<HTMLTextAreaElement | null>;
  onAnalyzeSubmit: (event: FormEvent<HTMLFormElement>, modelProfile: string) => void;
  onQuickPrompt: (prompt: string) => void;
  onCopyMessage: (message: ChatMessage) => void;
  onRegenerate: (messageId: number) => void;
  onFeedback: (messageId: number, value: "up" | "down") => void;
  onDeleteMessage: (messageId: number) => void;
  onOpenAttachment: (messageId: number) => void;
  onCopyCode: (content: string) => void;
  onVoiceInputNotice: () => void;
  onScrollToBottom: () => void;
  onRemoveComposerAttachment: () => void;
  onSymptomTextChange: (value: string) => void;
  onComposerKeyDown: (event: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  onAttachmentSelection: (file: File | null) => void;
  onToggleComposerDetails: () => void;
  onConsentChange: (value: boolean) => void;
  onSearchConsentChange: (value: boolean) => void;
  onSymptomTagsTextChange: (value: string) => void;
  onOpenGuidance: () => void;
  onOpenFacilities: () => void;
}

function renderInlineCode(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  let start = 0;
  INLINE_CODE_RE.lastIndex = 0;

  for (let match = INLINE_CODE_RE.exec(text); match !== null; match = INLINE_CODE_RE.exec(text)) {
    if (match.index > start) {
      nodes.push(<span key={`${keyPrefix}-txt-${start}`}>{text.slice(start, match.index)}</span>);
    }
    nodes.push(<code key={`${keyPrefix}-code-${match.index}`}>{match[1]}</code>);
    start = match.index + match[0].length;
  }

  if (start < text.length) {
    nodes.push(<span key={`${keyPrefix}-txt-${start}`}>{text.slice(start)}</span>);
  }
  if (!nodes.length) {
    nodes.push(<span key={`${keyPrefix}-plain`}>{text}</span>);
  }
  return nodes;
}

function renderMessageContent(entry: ChatMessage, onCopyCode: (content: string) => void): ReactNode {
  const segments = segmentMessageContent(entry.content || "");
  return (
    <>
      {segments.map((segment, idx) => {
        if (segment.type === "code") {
          return (
            <div key={`${entry.id}-code-${idx}`} className="code-block-shell">
              <div className="code-block-top">
                <span>{segment.language || "code"}</span>
                <button type="button" className="inline-action" onClick={() => onCopyCode(segment.content)}>
                  Copy
                </button>
              </div>
              <pre className="code-block">
                <code>{segment.content}</code>
              </pre>
            </div>
          );
        }
        const lines = segment.content
          .split("\n")
          .filter((line, lineIdx, arr) => line.trim() || lineIdx < arr.length - 1);
        if (!lines.length) return null;
        return (
          <div key={`${entry.id}-text-${idx}`} className="rich-text">
            {lines.map((line, lineIdx) => (
              <p key={`${entry.id}-line-${idx}-${lineIdx}`}>
                {renderInlineCode(line, `${entry.id}-${idx}-${lineIdx}`)}
              </p>
            ))}
          </div>
        );
      })}
    </>
  );
}

function AssistantAnalysisCard({
  analysis,
  conditions,
  onOpenFacilities,
}: {
  analysis: AnalysisResponse;
  conditions: Array<{ condition: string; probability: number }>;
  onOpenFacilities?: () => void;
}) {
  const isHigh = analysis.risk_level === "High";
  return (
    <div className="analysis-card">
      {/* REQ-8: Emergency auto-trigger banner */}
      {(isHigh || analysis.needs_urgent_care) && (
        <div
          className="emergency-banner"
          style={{
            background: "#fee2e2",
            border: "1px solid #ef4444",
            borderRadius: "6px",
            padding: "10px 14px",
            marginBottom: "10px",
            color: "#991b1b",
          }}
        >
          <strong>Emergency Alert:</strong> High-risk pattern detected. Seek immediate in-person care.
          {onOpenFacilities && (
            <button
              type="button"
              style={{ marginLeft: "10px", fontSize: "0.85rem", padding: "2px 10px" }}
              onClick={onOpenFacilities}
            >
              Find Emergency Facilities
            </button>
          )}
        </div>
      )}

      <p className={`risk-chip risk-${analysis.risk_level.toLowerCase()}`}>
        Risk: {analysis.risk_level} ({analysis.risk_score.toFixed(2)})
      </p>
      {conditions.length > 0 ? (
        <ul>
          {conditions.map((item) => (
            <li key={item.condition}>
              {item.condition} ({(item.probability * 100).toFixed(1)}%)
            </li>
          ))}
        </ul>
      ) : (
        <p>No specific named condition identified from current input.</p>
      )}
      <p>
        <strong>Next step:</strong> {analysis.recommendation_text}
      </p>
      {analysis.needs_urgent_care && <p className="urgent-note">Urgent warning: seek immediate in-person care.</p>}

      {/* REQ-3: Calibration disclaimer */}
      <p
        className="muted"
        style={{ fontSize: "0.78rem", marginTop: "8px", borderTop: "1px solid var(--border)", paddingTop: "6px" }}
      >
        Confidence scores are indicative only and not clinically calibrated. This is not a diagnosis.
        Always consult a qualified healthcare professional.
      </p>
    </div>
  );
}

export function ChatPane({
  selectedModel,
  visibleMessages,
  messageAttachments,
  feedbackByMessage,
  streamingMessageId,
  isTyping,
  showScrollToBottom,
  imagePreviewUrl,
  imageFile,
  busy,
  symptomText,
  symptomTagsText,
  consentGiven,
  searchConsentGiven,
  showComposerDetails,
  chatFeedRef,
  composerRef,
  onAnalyzeSubmit,
  onQuickPrompt,
  onCopyMessage,
  onRegenerate,
  onFeedback,
  onDeleteMessage,
  onOpenAttachment,
  onCopyCode,
  onVoiceInputNotice,
  onScrollToBottom,
  onRemoveComposerAttachment,
  onSymptomTextChange,
  onComposerKeyDown,
  onAttachmentSelection,
  onToggleComposerDetails,
  onConsentChange,
  onSearchConsentChange,
  onSymptomTagsTextChange,
  onOpenGuidance,
  onOpenFacilities,
}: ChatPaneProps) {
  return (
    <section className="chat-stage">
      {visibleMessages.length === 0 && (
        <section className="welcome-card">
          <h3>How can I help you today?</h3>
          <p>Describe symptoms in natural language. You can also attach an image or use quick prompts below.</p>
          <div className="quick-chip-row">
            {QUICK_PROMPTS.map((prompt) => (
              <button key={prompt} type="button" className="chip" onClick={() => onQuickPrompt(prompt)}>
                {prompt}
              </button>
            ))}
          </div>
        </section>
      )}

      <div ref={chatFeedRef as RefObject<HTMLDivElement>} className="message-stream">
        {visibleMessages.map((entry, index) => {
          const entryAnalysis = parseMessageAnalysis(entry);
          const conditions = entryAnalysis ? displayConditions(entryAnalysis, 3) : [];
          const groupStart = index === 0 || visibleMessages[index - 1].role !== entry.role;
          const stamp = shouldShowTimestamp(visibleMessages, index);
          const attachment = messageAttachments[entry.id];
          const showDayDivider =
            index === 0 || dayKey(visibleMessages[index - 1].created_at) !== dayKey(entry.created_at);

          return (
            <Fragment key={entry.id}>
              {showDayDivider && (
                <div className="day-divider">
                  <span>{formatDayLabel(entry.created_at)}</span>
                </div>
              )}
              <article
                className={[
                  "message-row",
                  entry.role,
                  groupStart ? "group-start" : "",
                  streamingMessageId === entry.id ? "streaming" : "",
                ]
                  .join(" ")
                  .trim()}
              >
                {entry.role === "assistant" && groupStart && <div className="message-avatar">AI</div>}
                <div className="bubble">
                  <div className="bubble-body">{renderMessageContent(entry, onCopyCode)}</div>

                  {attachment && (
                    <button type="button" className="attachment-bubble" onClick={() => onOpenAttachment(entry.id)}>
                      {attachment.type.startsWith("image/") && <img src={attachment.url} alt={attachment.name} />}
                      <div>
                        <strong>{attachment.name}</strong>
                        <small>{formatBytes(attachment.size)}</small>
                      </div>
                    </button>
                  )}

                  {entry.role === "assistant" && entryAnalysis && (
                    <AssistantAnalysisCard analysis={entryAnalysis} conditions={conditions} onOpenFacilities={onOpenFacilities} />
                  )}

                  <div className="bubble-footer">
                    {stamp && <time>{new Date(entry.created_at).toLocaleString()}</time>}
                    <div className="message-actions">
                      <button type="button" className="inline-action" onClick={() => onCopyMessage(entry)}>
                        Copy
                      </button>
                      {entry.role === "assistant" && (
                        <button type="button" className="inline-action" onClick={() => onRegenerate(entry.id)}>
                          Regenerate
                        </button>
                      )}
                      {entry.role === "assistant" && (
                        <>
                          <button
                            type="button"
                            className={feedbackByMessage[entry.id] === "up" ? "inline-action active" : "inline-action"}
                            onClick={() => onFeedback(entry.id, "up")}
                          >
                            Helpful
                          </button>
                          <button
                            type="button"
                            className={feedbackByMessage[entry.id] === "down" ? "inline-action active" : "inline-action"}
                            onClick={() => onFeedback(entry.id, "down")}
                          >
                            Not Helpful
                          </button>
                        </>
                      )}
                      <button type="button" className="inline-action" onClick={() => onDeleteMessage(entry.id)}>
                        Delete
                      </button>
                    </div>
                  </div>
                </div>
                {entry.role === "user" && groupStart && <div className="message-avatar user">You</div>}
              </article>
            </Fragment>
          );
        })}

        {isTyping && (
          <article className="message-row assistant typing-row">
            <div className="message-avatar">AI</div>
            <div className="bubble typing-bubble">
              <div className="typing-indicator" aria-label="Assistant typing">
                <span />
                <span />
                <span />
              </div>
              <p className="muted">Assistant is typing...</p>
            </div>
          </article>
        )}
      </div>

      {showScrollToBottom && (
        <button type="button" className="scroll-fab" onClick={onScrollToBottom}>
          Latest
        </button>
      )}

      <form className="composer" onSubmit={(event) => onAnalyzeSubmit(event, selectedModel)}>
        <input type="hidden" name="model_profile" value={selectedModel} />
        {imagePreviewUrl && (
          <div className="composer-attachment-preview">
            <img src={imagePreviewUrl} alt={imageFile?.name || "Attachment"} />
            <div>
              <strong>{imageFile?.name || "Attachment"}</strong>
              <small>{imageFile ? formatBytes(imageFile.size) : ""}</small>
            </div>
            <button type="button" className="inline-action" onClick={onRemoveComposerAttachment}>
              Remove
            </button>
          </div>
        )}

        <textarea
          ref={composerRef as RefObject<HTMLTextAreaElement>}
          rows={1}
          value={symptomText}
          onChange={(event) => onSymptomTextChange(event.target.value)}
          onKeyDown={onComposerKeyDown}
          placeholder="Message Healthcare Assistant..."
          required
        />

        <div className="composer-bar">
          <div className="composer-left">
            <label className="icon-btn attach-btn" title="Attach image">
              Attach
              <input
                type="file"
                accept="image/png,image/jpeg"
                onChange={(event) => onAttachmentSelection(event.target.files?.[0] || null)}
                hidden
              />
            </label>
            <button type="button" className="icon-btn" onClick={onVoiceInputNotice} title="Voice input">
              Mic
            </button>
            <button type="button" className="icon-btn" onClick={onToggleComposerDetails}>
              {showComposerDetails ? "Less" : "More"}
            </button>
          </div>

          <div className="composer-right">
            <label className="checkbox-line">
              <input type="checkbox" checked={consentGiven} onChange={(event) => onConsentChange(event.target.checked)} />
              I consent
            </label>
            <label className="checkbox-line">
              <input
                type="checkbox"
                checked={searchConsentGiven}
                onChange={(event) => onSearchConsentChange(event.target.checked)}
              />
              Allow redacted external search
            </label>
            <button type="submit" className="send-btn" disabled={busy}>
              {busy ? "Sending..." : "Send"}
            </button>
          </div>
        </div>

        {showComposerDetails && (
          <div className="composer-extra">
            <label>
              Symptom tags (comma separated)
              <input
                value={symptomTagsText}
                onChange={(event) => onSymptomTagsTextChange(event.target.value)}
                placeholder="fever, cough"
              />
            </label>
            <div className="extra-actions">
              <button type="button" className="ghost" onClick={onOpenGuidance}>
                Open Guidance
              </button>
              <button type="button" className="ghost" onClick={onOpenFacilities}>
                Open Facilities
              </button>
            </div>
          </div>
        )}
      </form>
    </section>
  );
}
