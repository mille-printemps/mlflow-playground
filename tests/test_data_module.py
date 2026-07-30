from unittest.mock import patch

from datasets import Dataset, DatasetDict

import data_module

CONFIG = {
    "model": {"name": "distilbert-base-uncased"},
    "data": {
        "dataset_name": "stanfordnlp/imdb",
        "train_subset_size": 4,
        "eval_subset_size": 3,
        "max_length": 8,
    },
    "training": {"seed": 42},
}


class FakeTokenizer:
    def __call__(self, texts, truncation=True, padding="max_length", max_length=8):
        return {
            "input_ids": [[1] * max_length for _ in texts],
            "attention_mask": [[1] * max_length for _ in texts],
        }


def make_fake_dataset_dict(n_train=6, n_test=6):
    train = Dataset.from_dict({
        "text": [f"train example {i}" for i in range(n_train)],
        "label": [i % 2 for i in range(n_train)],
    })
    test = Dataset.from_dict({
        "text": [f"test example {i}" for i in range(n_test)],
        "label": [i % 2 for i in range(n_test)],
    })
    return DatasetDict({"train": train, "test": test})


def test_load_and_tokenize_returns_expected_subset_sizes():
    fake_dataset = make_fake_dataset_dict()
    with patch.object(data_module, "load_dataset", return_value=fake_dataset), \
         patch.object(data_module.AutoTokenizer, "from_pretrained", return_value=FakeTokenizer()):
        train_tok, eval_tok, _ = data_module.load_and_tokenize(CONFIG)

    assert len(train_tok) == CONFIG["data"]["train_subset_size"]
    assert len(eval_tok) == CONFIG["data"]["eval_subset_size"]


def test_load_and_tokenize_removes_text_column_and_sets_torch_format():
    fake_dataset = make_fake_dataset_dict()
    with patch.object(data_module, "load_dataset", return_value=fake_dataset), \
         patch.object(data_module.AutoTokenizer, "from_pretrained", return_value=FakeTokenizer()):
        train_tok, eval_tok, _ = data_module.load_and_tokenize(CONFIG)

    for tokenized in (train_tok, eval_tok):
        assert "text" not in tokenized.column_names
        assert "input_ids" in tokenized.column_names
        assert "label" in tokenized.column_names
        assert tokenized.format["type"] == "torch"


def test_load_and_tokenize_uses_configured_model_name_for_tokenizer():
    fake_dataset = make_fake_dataset_dict()
    with patch.object(data_module, "load_dataset", return_value=fake_dataset), \
         patch.object(
             data_module.AutoTokenizer, "from_pretrained", return_value=FakeTokenizer()
         ) as mock_from_pretrained:
        data_module.load_and_tokenize(CONFIG)

    mock_from_pretrained.assert_called_once_with(CONFIG["model"]["name"])
