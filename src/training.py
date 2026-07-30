import evaluate
import mlflow
import numpy as np
import yaml
from transformers import Trainer, TrainingArguments

from data_module import load_and_tokenize
from model_module import get_device, load_model


def load_config(path: str = "../configs/train_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)

    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}


def main():
    config = load_config()

    device = get_device()
    print(f"Using device: {device}")

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    train_dataset, eval_dataset, tokenizer = load_and_tokenize(config)
    model = load_model(config)
    model.to(device)

    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=config["training"]["weight_decay"],
        logging_steps=config["training"]["logging_steps"],
        eval_strategy=config["training"]["eval_strategy"],
        save_strategy=config["training"]["save_strategy"],
        seed=config["training"]["seed"],
        report_to=[],  # we handle MLflow logging manually below
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run():
        # log config as params
        mlflow.log_params(
            {
                "model_name": config["model"]["name"],
                "train_subset_size": config["data"]["train_subset_size"],
                "epochs": config["training"]["num_train_epochs"],
                "batch_size": config["training"]["per_device_train_batch_size"],
                "learning_rate": config["training"]["learning_rate"],
                "device": str(device),
            }
        )

        trainer.train()

        eval_results = trainer.evaluate()
        mlflow.log_metrics(
            {
                "eval_accuracy": eval_results["eval_accuracy"],
                "eval_f1": eval_results["eval_f1"],
                "eval_loss": eval_results["eval_loss"],
            }
        )

        # log the model itself
        model_info = mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="model",
            task="text-classification",
        )

        # register it in the Model Registry
        registered_model_name = "sentiment-distilbert"
        mlflow.register_model(
            model_uri=model_info.model_uri,
            name=registered_model_name,
        )

        print(f"Model registered as '{registered_model_name}'")

        print(
            f"Run complete. Eval accuracy: {eval_results['eval_accuracy']:.4f}, "
            f"F1: {eval_results['eval_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
