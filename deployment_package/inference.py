"""
Healthcare AI Inference Module
Production-ready inference for all three models:
  1. Triage Text Classification (symptom → condition)
  2. Dialogue Intent Classification
  3. Skin Lesion Image Classification

Compatible with FastAPI and Django REST Framework.
"""
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import joblib

logger = logging.getLogger(__name__)

# ─── Model paths (relative to this file) ─────────────────────────────────────
_HERE = Path(__file__).parent
_MODELS = _HERE / "models"

# ─── Lazy-loaded model cache ──────────────────────────────────────────────────
_triage_model = None
_triage_vec = None
_triage_labels = None

_dialogue_model = None
_dialogue_vec = None
_dialogue_labels = None

_image_model = None
_image_labels = None
_image_meta = None


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1: TRIAGE TEXT CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def _load_triage():
    global _triage_model, _triage_vec, _triage_labels
    if _triage_model is None:
        _triage_model = joblib.load(_MODELS / "text_classifier.joblib")
        _triage_vec   = joblib.load(_MODELS / "tfidf_vectorizer.joblib")
        with open(_MODELS / "text_labels.json") as f:
            _triage_labels = json.load(f)
        logger.info(f"Triage model loaded: {len(_triage_labels)} classes")


def _truncate_text(text: str, max_words: int = 50) -> str:
    """Truncate to first max_words words (matches training preprocessing)."""
    words = str(text).split()
    return " ".join(words[:max_words])


def predict_triage(symptom_text: str, top_k: int = 5) -> Dict:
    """
    Predict probable conditions from symptom text.
    
    Args:
        symptom_text: Patient symptom description
        top_k: Number of top predictions to return
    
    Returns:
        {
            "predictions": [{"condition": str, "probability": float}, ...],
            "confidence": float,
            "model_version": str,
            "latency_ms": int
        }
    """
    _validate_text_input(symptom_text)
    _load_triage()
    
    t0 = time.monotonic()
    clean_text = _truncate_text(symptom_text)
    X = _triage_vec.transform([clean_text])
    
    # SGDClassifier with modified_huber supports predict_proba
    probs = _triage_model.predict_proba(X)[0]
    
    predictions = [
        {"condition": _triage_labels[i], "probability": round(float(p), 4)}
        for i, p in enumerate(probs)
    ]
    predictions.sort(key=lambda x: x["probability"], reverse=True)
    predictions = predictions[:top_k]
    
    latency_ms = int((time.monotonic() - t0) * 1000)
    confidence = predictions[0]["probability"] if predictions else 0.0
    
    return {
        "predictions": predictions,
        "confidence": round(confidence, 4),
        "model_version": "triage-sgd-v2",
        "latency_ms": latency_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2: DIALOGUE INTENT CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def _load_dialogue():
    global _dialogue_model, _dialogue_vec, _dialogue_labels
    if _dialogue_model is None:
        _dialogue_model = joblib.load(_MODELS / "dialogue_intent_classifier.joblib")
        _dialogue_vec   = joblib.load(_MODELS / "dialogue_intent_vectorizer.joblib")
        with open(_MODELS / "dialogue_intent_labels.json") as f:
            _dialogue_labels = json.load(f)
        logger.info(f"Dialogue model loaded: {len(_dialogue_labels)} intents")


def predict_intent(text: str) -> Dict:
    """
    Classify the intent of a medical dialogue message.
    
    Args:
        text: User message text
    
    Returns:
        {
            "intent": str,
            "confidence": float,
            "all_intents": [{"intent": str, "probability": float}, ...],
            "model_version": str,
            "latency_ms": int
        }
    """
    _validate_text_input(text)
    _load_dialogue()
    
    t0 = time.monotonic()
    X = _dialogue_vec.transform([str(text)])
    
    probs = _dialogue_model.predict_proba(X)[0]
    best_idx = int(np.argmax(probs))
    
    all_intents = [
        {"intent": _dialogue_labels[i], "probability": round(float(p), 4)}
        for i, p in enumerate(probs)
    ]
    all_intents.sort(key=lambda x: x["probability"], reverse=True)
    
    latency_ms = int((time.monotonic() - t0) * 1000)
    
    return {
        "intent": _dialogue_labels[best_idx],
        "confidence": round(float(probs[best_idx]), 4),
        "all_intents": all_intents[:5],
        "model_version": "dialogue-sgd-v2",
        "latency_ms": latency_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3: SKIN LESION IMAGE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def _load_image():
    global _image_model, _image_labels, _image_meta
    if _image_model is None:
        try:
            import torch
            checkpoint = torch.load(_MODELS / "skin_cnn_torch.pt", map_location="cpu")
            
            # Build model
            num_classes = int(checkpoint.get("num_classes", 7))
            architecture = str(checkpoint.get("architecture", "improved_dermcnn"))
            
            if architecture == "improved_dermcnn":
                import torch.nn as nn
                
                class ImprovedDermCNN(nn.Module):
                    def __init__(self, num_classes=7):
                        super().__init__()
                        self.features = nn.Sequential(
                            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
                            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
                            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                            nn.MaxPool2d(2, 2), nn.Dropout2d(0.3),
                        )
                        self.classifier = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(128 * 3 * 3, 256), nn.ReLU(inplace=True),
                            nn.Dropout(0.4), nn.Linear(256, num_classes),
                        )
                    def forward(self, x):
                        return self.classifier(self.features(x))
                
                model = ImprovedDermCNN(num_classes=num_classes)
            else:
                # Fallback: try torchvision
                import torchvision.models as tv_models
                import torch.nn as nn
                model = tv_models.efficientnet_b0(weights=None)
                in_features = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.3, inplace=True),
                    nn.Linear(in_features, num_classes),
                )
            
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            
            _image_model = {"model": model, "checkpoint": checkpoint}
            _image_labels = checkpoint.get("class_names", [f"class_{i}" for i in range(num_classes)])
            _image_meta = {
                "input_size": int(checkpoint.get("input_size", 28)),
                "normalization": checkpoint.get("normalization", {}),
                "architecture": architecture,
            }
            logger.info(f"Image model loaded: {architecture}, {num_classes} classes")
        except Exception as e:
            logger.error(f"Failed to load image model: {e}")
            _image_model = None


def predict_image(image_path: Union[str, Path], top_k: int = 5) -> Dict:
    """
    Classify a skin lesion image.
    
    Args:
        image_path: Path to image file (JPEG/PNG)
        top_k: Number of top predictions to return
    
    Returns:
        {
            "predictions": [{"condition": str, "probability": float}, ...],
            "confidence": float,
            "model_version": str,
            "latency_ms": int
        }
    """
    _load_image()
    
    if _image_model is None:
        return {
            "predictions": [],
            "confidence": 0.0,
            "model_version": "image-unavailable",
            "latency_ms": 0,
            "error": "Image model not available",
        }
    
    t0 = time.monotonic()
    
    try:
        import torch
        from PIL import Image
        
        input_size = _image_meta["input_size"]
        norm = _image_meta.get("normalization", {})
        mean = norm.get("mean", [0.5, 0.5, 0.5])
        std  = norm.get("std",  [0.2, 0.2, 0.2])
        
        img = Image.open(image_path).convert("RGB").resize((input_size, input_size))
        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0)
        
        mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std_t  = torch.tensor(std,  dtype=torch.float32).view(1, 3, 1, 1)
        img_t = (img_t - mean_t) / (std_t + 1e-6)
        
        with torch.no_grad():
            logits = _image_model["model"](img_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        predictions = [
            {"condition": _image_labels[i], "probability": round(float(p), 4)}
            for i, p in enumerate(probs)
        ]
        predictions.sort(key=lambda x: x["probability"], reverse=True)
        predictions = predictions[:top_k]
        
        latency_ms = int((time.monotonic() - t0) * 1000)
        confidence = predictions[0]["probability"] if predictions else 0.0
        
        return {
            "predictions": predictions,
            "confidence": round(confidence, 4),
            "model_version": f"image-{_image_meta['architecture']}-v2",
            "latency_ms": latency_ms,
        }
    except Exception as e:
        logger.error(f"Image inference error: {e}")
        return {
            "predictions": [],
            "confidence": 0.0,
            "model_version": "image-error",
            "latency_ms": int((time.monotonic() - t0) * 1000),
            "error": str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# INPUT VALIDATION & DRIFT DETECTION STUBS
# ─────────────────────────────────────────────────────────────────────────────

def _validate_text_input(text: str) -> None:
    """Validate text input for safety and quality."""
    if not text or not str(text).strip():
        raise ValueError("Input text cannot be empty")
    if len(str(text)) > 10000:
        raise ValueError("Input text too long (max 10,000 characters)")


def check_input_drift(text: str) -> Dict:
    """
    Stub for input drift detection.
    In production: compare text statistics against training distribution.
    """
    words = str(text).split()
    return {
        "word_count": len(words),
        "char_count": len(text),
        "drift_detected": False,  # Stub — implement with reference statistics
        "drift_score": 0.0,
    }


def health_check() -> Dict:
    """Health check endpoint for monitoring."""
    status = {}
    
    try:
        _load_triage()
        status["triage"] = {"status": "ok", "classes": len(_triage_labels)}
    except Exception as e:
        status["triage"] = {"status": "error", "error": str(e)}
    
    try:
        _load_dialogue()
        status["dialogue"] = {"status": "ok", "intents": len(_dialogue_labels)}
    except Exception as e:
        status["dialogue"] = {"status": "error", "error": str(e)}
    
    try:
        _load_image()
        status["image"] = {"status": "ok" if _image_model else "unavailable"}
    except Exception as e:
        status["image"] = {"status": "error", "error": str(e)}
    
    return {"status": "healthy" if all(v.get("status") == "ok" for v in status.values()) else "degraded",
            "models": status}


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app (optional — run standalone)
# ─────────────────────────────────────────────────────────────────────────────

def create_fastapi_app():
    """Create a FastAPI app for standalone deployment."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("Install fastapi and pydantic: pip install fastapi uvicorn")
    
    app = FastAPI(title="Healthcare AI Inference API", version="2.0.0")
    
    class TriageRequest(BaseModel):
        symptom_text: str
        top_k: int = 5
    
    class IntentRequest(BaseModel):
        text: str
    
    @app.get("/health")
    def health():
        return health_check()
    
    @app.post("/predict/triage")
    def triage(req: TriageRequest):
        try:
            return predict_triage(req.symptom_text, top_k=req.top_k)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/predict/intent")
    def intent(req: IntentRequest):
        try:
            return predict_intent(req.text)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/predict/image")
    def image(image_path: str, top_k: int = 5):
        return predict_image(image_path, top_k=top_k)
    
    return app


if __name__ == "__main__":
    import uvicorn
    app = create_fastapi_app()
    uvicorn.run(app, host="0.0.0.0", port=8080)
