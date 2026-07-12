import os
import mlflow
import mlflow.transformers
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Sentiment Model Serving")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "sentiment-distilbert")
MODEL_STAGE_OR_VERSION = os.getenv("MODEL_VERSION", "latest")  # or a specific version number

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load the model once at startup, not per-request
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


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    result = model_pipeline(request.text)[0]
    return PredictResponse(label=result["label"], score=result["score"])
