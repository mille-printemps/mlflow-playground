import os
import time
import json
import logging
from datetime import datetime, timezone

import mlflow
import mlflow.transformers
from fastapi import FastAPI, Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="Sentiment Model Serving")

REQUEST_COUNT = Counter(
    "predict_requests_total", "Total number of /predict requests", ["status"]
)
REQUEST_LATENCY = Histogram(
    "predict_latency_seconds", "Latency of /predict requests in seconds"
)
PREDICTION_LABEL_COUNT = Counter(
    "predict_labels_total", "Total predictions by label", ["label"]
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "sentiment-distilbert")
MODEL_STAGE_OR_VERSION = os.getenv("MODEL_VERSION", "latest")
LOG_PATH = os.getenv("PREDICTION_LOG_PATH", "/app/logs/predictions.jsonl")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Set up a dedicated logger that writes structured JSON lines to a file
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
prediction_logger = logging.getLogger("prediction_logger")
prediction_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_PATH)
file_handler.setFormatter(logging.Formatter("%(message)s"))
prediction_logger.addHandler(file_handler)

if MODEL_STAGE_OR_VERSION == "latest":
    model_uri = f"models:/{MODEL_NAME}/latest"
else:
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE_OR_VERSION}"

print(f"Loading model from: {model_uri}")
model_pipeline = mlflow.transformers.load_model(model_uri, return_type="pipeline")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    score: float


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "version": MODEL_STAGE_OR_VERSION}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start_time = time.time()
    try:
        result = model_pipeline(request.text)[0]
    except Exception:
        REQUEST_COUNT.labels(status="error").inc()
        raise
    latency_s = time.time() - start_time

    REQUEST_COUNT.labels(status="success").inc()
    REQUEST_LATENCY.observe(latency_s)
    PREDICTION_LABEL_COUNT.labels(label=result["label"]).inc()

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": request.text,
        "text_length": len(request.text),
        "predicted_label": result["label"],
        "score": result["score"],
        "latency_ms": round(latency_s * 1000, 2),
        "model_name": MODEL_NAME,
        "model_version": MODEL_STAGE_OR_VERSION,
    }
    prediction_logger.info(json.dumps(log_entry))

    return PredictResponse(label=result["label"], score=result["score"])