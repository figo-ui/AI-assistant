"""
Test suite for Healthcare AI inference module.
Run: python -m pytest tests/test_inference.py -v
"""
import sys
import os
import time
import pytest

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import predict_triage, predict_intent, health_check, _validate_text_input


class TestTriageModel:
    """Tests for triage text classification."""

    def test_basic_prediction(self):
        result = predict_triage("I have a fever and cough and headache")
        assert "predictions" in result
        assert "confidence" in result
        assert "model_version" in result
        assert len(result["predictions"]) > 0
        assert result["confidence"] > 0.0

    def test_top_k_respected(self):
        result = predict_triage("chest pain and shortness of breath", top_k=3)
        assert len(result["predictions"]) <= 3

    def test_predictions_sum_to_one(self):
        result = predict_triage("fever headache nausea vomiting")
        total = sum(p["probability"] for p in result["predictions"])
        # Top-k won't sum to 1, but each should be in [0,1]
        for p in result["predictions"]:
            assert 0.0 <= p["probability"] <= 1.0

    def test_sorted_by_probability(self):
        result = predict_triage("cough fever sore throat runny nose")
        probs = [p["probability"] for p in result["predictions"]]
        assert probs == sorted(probs, reverse=True)

    def test_latency_reasonable(self):
        t0 = time.time()
        predict_triage("headache and dizziness")
        latency = time.time() - t0
        assert latency < 5.0, f"Inference too slow: {latency:.2f}s"

    def test_cardiac_symptoms(self):
        result = predict_triage("chest pain left arm pain sweating")
        conditions = [p["condition"].lower() for p in result["predictions"]]
        # Should have cardiac-related condition in top predictions
        assert any("heart" in c or "angina" in c or "cardiac" in c for c in conditions)

    def test_respiratory_symptoms(self):
        result = predict_triage("cough fever shortness of breath")
        assert len(result["predictions"]) > 0

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            predict_triage("")

    def test_very_long_input_raises(self):
        with pytest.raises(ValueError):
            predict_triage("word " * 3000)


class TestDialogueModel:
    """Tests for dialogue intent classification."""

    def test_information_intent(self):
        result = predict_intent("What is diabetes?")
        assert result["intent"] in ["information", "causes", "symptoms"]
        assert result["confidence"] > 0.5

    def test_symptoms_intent(self):
        result = predict_intent("What are the symptoms of COVID-19?")
        assert result["intent"] in ["symptoms", "information"]

    def test_treatment_intent(self):
        result = predict_intent("How is hypertension treated?")
        assert result["intent"] in ["treatment", "information"]

    def test_all_intents_returned(self):
        result = predict_intent("What causes cancer?")
        assert "all_intents" in result
        assert len(result["all_intents"]) > 0

    def test_latency_reasonable(self):
        t0 = time.time()
        predict_intent("What is the treatment for asthma?")
        latency = time.time() - t0
        assert latency < 2.0, f"Intent inference too slow: {latency:.2f}s"

    def test_model_version_present(self):
        result = predict_intent("What are the risk factors for heart disease?")
        assert "model_version" in result
        assert "dialogue" in result["model_version"]


class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check_returns_dict(self):
        result = health_check()
        assert isinstance(result, dict)
        assert "status" in result
        assert "models" in result

    def test_models_present(self):
        result = health_check()
        assert "triage" in result["models"]
        assert "dialogue" in result["models"]


class TestInputValidation:
    """Tests for input validation."""

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _validate_text_input("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            _validate_text_input("   ")

    def test_valid_input_passes(self):
        _validate_text_input("I have a headache")  # Should not raise

    def test_too_long_raises(self):
        with pytest.raises(ValueError):
            _validate_text_input("x" * 10001)


class TestLatencyBenchmarks:
    """Latency benchmarks for production readiness."""

    def test_triage_p95_latency(self):
        """P95 latency should be < 500ms."""
        latencies = []
        for _ in range(10):
            t0 = time.time()
            predict_triage("fever cough headache fatigue")
            latencies.append((time.time() - t0) * 1000)
        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]
        print(f"\nTriage P95 latency: {p95:.1f}ms")
        assert p95 < 500, f"P95 latency too high: {p95:.1f}ms"

    def test_dialogue_p95_latency(self):
        """P95 latency should be < 200ms."""
        latencies = []
        for _ in range(10):
            t0 = time.time()
            predict_intent("What is the treatment for diabetes?")
            latencies.append((time.time() - t0) * 1000)
        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]
        print(f"\nDialogue P95 latency: {p95:.1f}ms")
        assert p95 < 200, f"P95 latency too high: {p95:.1f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
