import type { AnalysisResponse, ChatMessage } from "../../types/analysis";

export const GENERIC_CONDITION_RE = /^(condition\s+\d+|class_\d+)$/i;
const CODE_BLOCK_RE = /```([\w+-]*)\n?([\s\S]*?)```/g;

export const QUICK_PROMPTS = [
  "I have fever and body aches since yesterday.",
  "I have lower back pain and mild fever for several days.",
  "I have cough, sore throat, and fatigue.",
  "Help me find a nearby clinic.",
];

export interface ContentSegment {
  type: "text" | "code";
  content: string;
  language?: string;
}

export interface LocalAttachment {
  name: string;
  url: string;
  type: string;
  size: number;
}

export function latestAnalysis(messages: ChatMessage[]): AnalysisResponse | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const entry = messages[index];
    const result = (entry.metadata as Record<string, unknown>)?.result;
    if (entry.role === "assistant" && result && typeof result === "object") {
      return result as AnalysisResponse;
    }
  }
  return null;
}

export function parseMessageAnalysis(message: ChatMessage): AnalysisResponse | null {
  const result = (message.metadata as Record<string, unknown>)?.result;
  if (!result || typeof result !== "object") {
    return null;
  }
  const typed = result as AnalysisResponse;
  if (!Array.isArray(typed.probable_conditions)) {
    return null;
  }
  return typed;
}

export function cleanConditionName(raw: string): string {
  const text = raw.replace(/\(disorder\)/gi, "").trim();
  if (!text || GENERIC_CONDITION_RE.test(text)) {
    return "Unspecified clinical pattern";
  }
  return text;
}

export function displayConditions(
  analysis: AnalysisResponse,
  limit = 3,
): Array<{ condition: string; probability: number }> {
  const source = Array.isArray(analysis.probable_conditions) ? analysis.probable_conditions : [];
  const normalized = source.map((item) => ({
    condition: cleanConditionName(String(item.condition || "")),
    probability: Number(item.probability || 0),
  }));

  const named = normalized.filter((item) => item.condition !== "Unspecified clinical pattern");
  if (named.length > 0) {
    return named.slice(0, limit);
  }
  return [];
}

export function segmentMessageContent(content: string): ContentSegment[] {
  const segments: ContentSegment[] = [];
  let startIndex = 0;
  CODE_BLOCK_RE.lastIndex = 0;

  for (let match = CODE_BLOCK_RE.exec(content); match !== null; match = CODE_BLOCK_RE.exec(content)) {
    if (match.index > startIndex) {
      segments.push({ type: "text", content: content.slice(startIndex, match.index) });
    }
    segments.push({
      type: "code",
      language: (match[1] || "").trim().toLowerCase(),
      content: match[2] || "",
    });
    startIndex = match.index + match[0].length;
  }

  if (startIndex < content.length) {
    segments.push({ type: "text", content: content.slice(startIndex) });
  }

  if (!segments.length) {
    return [{ type: "text", content }];
  }
  return segments;
}

export function shouldShowTimestamp(entries: ChatMessage[], index: number): boolean {
  if (index === entries.length - 1) return true;
  const current = entries[index];
  const next = entries[index + 1];
  if (current.role !== next.role) return true;
  const currentTime = new Date(current.created_at).getTime();
  const nextTime = new Date(next.created_at).getTime();
  return nextTime - currentTime > 2 * 60 * 1000;
}

export function dayKey(value: string): string {
  const date = new Date(value);
  return `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`;
}

export function formatDayLabel(value: string): string {
  const date = new Date(value);
  return date.toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}
