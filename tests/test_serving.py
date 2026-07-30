import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

SERVING_DIR = Path(__file__).resolve().parents[1] / "serving"


@pytest.fixture(scope="module")
def app_module(tmp_path_factory):
    log_path = tmp_path_factory.mktemp("logs") / "predictions.jsonl"
    env = {
        "MLFLOW_TRACKING_URI": "http://127.0.0.1:5000",
        "MODEL_NAME": "sentiment-distilbert",
        "MODEL_VERSION": "latest",
        "PREDICTION_LOG_PATH": str(log_path),
    }
    with patch.dict("os.environ", env), \
         patch("mlflow.transformers.load_model", return_value=MagicMock()):
        sys.modules.pop("app", None)
        import app as module
        yield module
        sys.modules.pop("app", None)


@pytest.fixture(scope="module")
def client(app_module):
    return TestClient(app_module.app, raise_server_exceptions=False)


def test_health_returns_model_info(client, app_module):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "model": app_module.MODEL_NAME,
        "version": app_module.MODEL_STAGE_OR_VERSION,
    }


def test_predict_returns_pipeline_result_and_logs_it(client, app_module):
    app_module.model_pipeline.return_value = [{"label": "positive", "score": 0.987}]

    resp = client.post("/predict", json={"text": "great movie"})

    assert resp.status_code == 200
    assert resp.json() == {"label": "positive", "score": 0.987}

    logged_lines = Path(app_module.LOG_PATH).read_text().strip().splitlines()
    last_entry = json.loads(logged_lines[-1])
    assert last_entry["text"] == "great movie"
    assert last_entry["predicted_label"] == "positive"
    assert last_entry["score"] == 0.987
    assert "latency_ms" in last_entry


def test_predict_failure_returns_500_and_counts_as_error(client, app_module):
    app_module.model_pipeline.side_effect = RuntimeError("boom")

    resp = client.post("/predict", json={"text": "x"})
    assert resp.status_code == 500

    app_module.model_pipeline.side_effect = None


def test_metrics_endpoint_exposes_request_counters(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "predict_requests_total" in resp.text
    assert 'status="success"' in resp.text
    assert 'status="error"' in resp.text
