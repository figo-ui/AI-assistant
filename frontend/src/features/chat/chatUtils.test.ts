import { describe, expect, it } from "vitest";

import type { AnalysisResponse, ChatMessage } from "../../types/analysis";
import {
  cleanConditionName,
  displayConditions,
  latestAnalysis,
  parseMessageAnalysis,
} from "./chatUtils";

function buildAnalysis(overrides: Partial<AnalysisResponse> = {}): AnalysisResponse {
  return {
    case_id: 1,
    created_at: "2026-03-08T10:00:00Z",
    probable_conditions: [],
    risk_score: 0.4,
    risk_level: "Medium",
    confidence_band: "medium",
    recommendation_text: "Schedule a clinic visit.",
    prevention_advice: [],
    disclaimer_text: "Demo only.",
    risk_factors: [],
    red_flags: [],
    needs_urgent_care: false,
    response_format_sections: [],
    clinical_report: {},
    modality_predictions: {
      text: [],
      image: [],
    },
    nearby_facilities: [],
    model_versions: {
      text: "text-v1",
      image: "image-v1",
      fusion: "fusion-v1",
    },
    latency_ms: 42,
    ...overrides,
  };
}

describe("chat utils", () => {
  it("normalizes generic condition labels", () => {
    expect(cleanConditionName("Condition 21")).toBe("Unspecified clinical pattern");
    expect(cleanConditionName("  Pneumonia (disorder) ")).toBe("Pneumonia");
  });

  it("returns only named conditions for display", () => {
    const analysis = buildAnalysis({
      probable_conditions: [
        { condition: "Condition 5", probability: 0.8 },
        { condition: "Pneumonia (disorder)", probability: 0.6 },
        { condition: "Common Cold", probability: 0.3 },
      ],
    });

    expect(displayConditions(analysis)).toEqual([
      { condition: "Pneumonia", probability: 0.6 },
      { condition: "Common Cold", probability: 0.3 },
    ]);
  });

  it("parses assistant metadata into analysis and finds the latest one", () => {
    const analysis = buildAnalysis({
      probable_conditions: [{ condition: "Stroke", probability: 0.72 }],
    });
    const messages: ChatMessage[] = [
      {
        id: 1,
        session: 1,
        role: "user",
        content: "hello",
        metadata: {},
        created_at: "2026-03-08T10:00:00Z",
      },
      {
        id: 2,
        session: 1,
        role: "assistant",
        content: "analysis",
        metadata: { result: analysis },
        created_at: "2026-03-08T10:01:00Z",
      },
    ];

    expect(parseMessageAnalysis(messages[1])).toEqual(analysis);
    expect(latestAnalysis(messages)).toEqual(analysis);
  });
});
