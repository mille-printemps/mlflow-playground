from datasets import load_dataset
from transformers import AutoTokenizer


def load_and_tokenize(config: dict):
    """Load IMDB dataset and tokenize it according to config."""
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    dataset = load_dataset(config["data"]["dataset_name"])

    train_subset = (
        dataset["train"]
        .shuffle(seed=config["training"]["seed"])
        .select(range(config["data"]["train_subset_size"]))
    )
    eval_subset = (
        dataset["test"]
        .shuffle(seed=config["training"]["seed"])
        .select(range(config["data"]["eval_subset_size"]))
    )

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=config["data"]["max_length"],
        )

    train_tokenized = train_subset.map(tokenize_fn, batched=True)
    eval_tokenized = eval_subset.map(tokenize_fn, batched=True)

    train_tokenized = train_tokenized.remove_columns(["text"])
    eval_tokenized = eval_tokenized.remove_columns(["text"])
    train_tokenized.set_format("torch")
    eval_tokenized.set_format("torch")

    return train_tokenized, eval_tokenized, tokenizer
