import torch
from transformers import AutoModelForSequenceClassification


def get_device():
    """Portable device selection: MPS (Mac) -> CUDA (Runpod later) -> CPU fallback."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(config: dict):
    id2label = {int(k): v for k, v in config["model"]["labels"].items()}
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["name"],
        num_labels=config["model"]["num_labels"],
        id2label=id2label,
        label2id=label2id,
    )
    return model
