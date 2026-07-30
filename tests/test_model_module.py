from unittest.mock import MagicMock, patch

import torch

import model_module

CONFIG = {"model": {"name": "distilbert-base-uncased", "num_labels": 2}}


def test_get_device_prefers_mps_when_available():
    with (
        patch.object(torch.backends.mps, "is_available", return_value=True),
        patch.object(torch.cuda, "is_available", return_value=True),
    ):
        assert model_module.get_device() == torch.device("mps")


def test_get_device_prefers_cuda_over_cpu_when_mps_unavailable():
    with (
        patch.object(torch.backends.mps, "is_available", return_value=False),
        patch.object(torch.cuda, "is_available", return_value=True),
    ):
        assert model_module.get_device() == torch.device("cuda")


def test_get_device_falls_back_to_cpu():
    with (
        patch.object(torch.backends.mps, "is_available", return_value=False),
        patch.object(torch.cuda, "is_available", return_value=False),
    ):
        assert model_module.get_device() == torch.device("cpu")


def test_load_model_configures_labels_and_num_labels():
    fake_model = MagicMock()
    with patch.object(
        model_module.AutoModelForSequenceClassification,
        "from_pretrained",
        return_value=fake_model,
    ) as mock_from_pretrained:
        result = model_module.load_model(CONFIG)

    assert result is fake_model
    mock_from_pretrained.assert_called_once_with(
        "distilbert-base-uncased",
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )
