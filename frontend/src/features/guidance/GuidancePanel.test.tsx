import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { AnalysisResponse, ChatMessage } from "../../types/analysis";
import { GuidancePanel } from "./GuidancePanel";

function buildAnalysis(overrides: Partial<AnalysisResponse> = {}): AnalysisResponse {
  return {
    case_id: 1,
    created_at: "2026-03-08T10:00:00Z",
    probable_conditions: [
      { condition: "Pneumonia", probability: 0.62 },
      { condition: "Common Cold", probability: 0.21 },
    ],
    risk_score: 0.55,
    risk_level: "Medium",
    confidence_band: "medium",
    recommendation_text: "Arrange a same-day assessment.",
    prevention_advice: ["Stay hydrated"],
    disclaimer_text: "Support tool only.",
    risk_factors: ["Fever and chest pain increase concern."],
    red_flags: ["shortness of breath"],
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
    latency_ms: 100,
    ...overrides,
  };
}

describe("GuidancePanel", () => {
  it("renders analysis details and recent assistant updates", () => {
    const recentAssistantMessages: ChatMessage[] = [
      {
        id: 99,
        session: 1,
        role: "assistant",
        content: "Monitor for worsening breathlessness.",
        metadata: {},
        created_at: "2026-03-08T10:15:00Z",
      },
    ];

    render(<GuidancePanel analysis={buildAnalysis()} recentAssistantMessages={recentAssistantMessages} />);

    expect(screen.getByText(/Guidance Summary/i)).toBeInTheDocument();
    expect(screen.getByText(/Arrange a same-day assessment/i)).toBeInTheDocument();
    expect(screen.getByText(/Pneumonia/i)).toBeInTheDocument();
    expect(screen.getByText(/Monitor for worsening breathlessness/i)).toBeInTheDocument();
  });

  it("shows the empty state when there is no analysis yet", () => {
    render(<GuidancePanel analysis={null} recentAssistantMessages={[]} />);

    expect(screen.getByText(/No analysis yet/i)).toBeInTheDocument();
    expect(screen.getByText(/No assistant updates yet/i)).toBeInTheDocument();
  });
});
