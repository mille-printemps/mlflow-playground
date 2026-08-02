from unittest.mock import MagicMock, patch

import pytest
import tools


def test_classify_sentiment_calls_serving_endpoint_and_formats_result():
    fake_response = MagicMock()
    fake_response.json.return_value = {"label": "positive", "score": 0.9871}
    fake_response.raise_for_status.return_value = None

    with patch.object(tools.httpx, "post", return_value=fake_response) as mock_post:
        result = tools.classify_sentiment.invoke({"text": "great movie"})

    mock_post.assert_called_once_with(
        f"{tools.SERVING_URL}/predict", json={"text": "great movie"}, timeout=10.0
    )
    assert result == "label=positive confidence=0.987"


def test_classify_sentiment_raises_on_http_error():
    fake_response = MagicMock()
    fake_response.raise_for_status.side_effect = tools.httpx.HTTPStatusError(
        "error", request=MagicMock(), response=MagicMock()
    )

    with patch.object(tools.httpx, "post", return_value=fake_response):
        with pytest.raises(tools.httpx.HTTPStatusError):
            tools.classify_sentiment.invoke({"text": "x"})
