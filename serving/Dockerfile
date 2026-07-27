FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# Install the CPU-only torch wheel first so the CUDA/nvidia-* deps never get pulled in by requirements.txt.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.2.0" \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000
ENV MODEL_NAME=sentiment-distilbert
ENV MODEL_VERSION=latest

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
