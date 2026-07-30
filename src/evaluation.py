import argparse
import statistics
import time

import mlflow
import mlflow.transformers
import numpy as np
import yaml
from datasets import load_dataset
from evaluate import load as load_metric

LABEL2ID = {"negative": 0, "positive": 1}


def load_config(path: str = "../configs/train_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_eval_examples(config: dict):
    """Pull the same eval split/seed used at training time, as raw text (no tokenization)."""
    dataset = load_dataset(config["data"]["dataset_name"])
    eval_subset = dataset["test"].shuffle(seed=config["training"]["seed"]).select(
        range(config["data"]["eval_subset_size"])
    )
    return eval_subset["text"], eval_subset["label"]


def load_registered_pipeline(model_name: str, model_version: str, tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}/{model_version}"
    print(f"Loading model from: {model_uri}")
    return mlflow.transformers.load_model(model_uri, return_type="pipeline")


def compute_accuracy_f1(pipeline, texts, labels, max_length: int):
    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")

    predictions = [
        LABEL2ID[pipeline(text, truncation=True, max_length=max_length)[0]["label"].lower()]
        for text in texts
    ]

    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    return acc["accuracy"], f1["f1"]


def measure_latency(pipeline, texts, max_length: int, num_samples: int):
    sample_texts = texts[:num_samples]
    latencies_ms = []
    for text in sample_texts:
        start = time.perf_counter()
        pipeline(text, truncation=True, max_length=max_length)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    return {
        "mean_latency_ms": round(statistics.mean(latencies_ms), 2),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "latency_samples": len(latencies_ms),
    }


def measure_model_size(pipeline):
    model = pipeline.model
    num_parameters = sum(p.numel() for p in model.parameters())
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return {
        "num_parameters": num_parameters,
        "model_size_mb": round(size_bytes / (1024 ** 2), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a registered sentiment model.")
    parser.add_argument("--config", default="../configs/train_config.yaml")
    parser.add_argument("--model-name", default="sentiment-distilbert")
    parser.add_argument("--model-version", default="latest")
    parser.add_argument("--max-eval-samples", type=int, default=200,
                         help="Cap on examples used for accuracy/F1 (eval set can be large).")
    parser.add_argument("--latency-samples", type=int, default=50)
    args = parser.parse_args()

    config = load_config(args.config)
    max_length = config["data"]["max_length"]

    print(f"Loading eval split from '{config['data']['dataset_name']}'...")
    texts, labels = load_eval_examples(config)

    pipeline = load_registered_pipeline(
        args.model_name, args.model_version, config["mlflow"]["tracking_uri"]
    )

    eval_texts = texts[: args.max_eval_samples]
    eval_labels = labels[: args.max_eval_samples]

    print(f"Scoring accuracy/F1 on {len(eval_texts)} examples...")
    accuracy, f1 = compute_accuracy_f1(pipeline, eval_texts, eval_labels, max_length)

    print(f"Measuring latency over {args.latency_samples} inferences...")
    latency_stats = measure_latency(pipeline, texts, max_length, args.latency_samples)

    size_stats = measure_model_size(pipeline)

    results = {
        "eval_accuracy": accuracy,
        "eval_f1": f1,
        **latency_stats,
        **size_stats,
    }

    print("\n--- Evaluation Results ---")
    for key, value in results.items():
        print(f"{key}: {value}")

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=f"eval-{args.model_name}-{args.model_version}"):
        mlflow.log_params({
            "model_name": args.model_name,
            "model_version": args.model_version,
            "eval_samples": len(eval_texts),
        })
        mlflow.log_metrics(results)

    print("\nLogged evaluation run to MLflow.")


if __name__ == "__main__":
    main()
