import os

import httpx
from langchain_core.tools import tool

SERVING_URL = os.getenv("SERVING_URL", "http://127.0.0.1:8000")


@tool
def classify_sentiment(text: str) -> str:
    """Classify the sentiment of a piece of text as positive or negative, with a confidence score.

    Use this whenever the user shares a review, comment, or message and asks about its
    sentiment, tone, or opinion, rather than guessing yourself.
    """
    response = httpx.post(f"{SERVING_URL}/predict", json={"text": text}, timeout=10.0)
    response.raise_for_status()
    result = response.json()
    return f"label={result['label']} confidence={result['score']:.3f}"
