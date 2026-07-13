import json
import argparse
import pandas as pd
from scipy import stats


def load_prediction_logs(log_path: str) -> pd.DataFrame:
    records = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def load_training_text_lengths(dataset_name: str = "stanfordnlp/imdb", subset_size: int = 3000) -> pd.Series:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name)
    train_subset = dataset["train"].shuffle(seed=42).select(range(subset_size))
    lengths = [len(t) for t in train_subset["text"]]
    return pd.Series(lengths)


def check_drift(training_lengths: pd.Series, production_lengths: pd.Series, alpha: float = 0.05):
    """Two-sample Kolmogorov-Smirnov test comparing text length distributions."""
    statistic, p_value = stats.ks_2samp(training_lengths, production_lengths)

    drift_detected = p_value < alpha

    return {
        "ks_statistic": round(statistic, 4),
        "p_value": round(p_value, 4),
        "drift_detected": drift_detected,
        "training_mean_length": round(training_lengths.mean(), 1),
        "production_mean_length": round(production_lengths.mean(), 1),
        "training_samples": len(training_lengths),
        "production_samples": len(production_lengths),
    }


def main():
    parser = argparse.ArgumentParser(description="Check for data drift between training and production text.")
    parser.add_argument("--log-path", default="../serving/logs/predictions.jsonl")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold for drift detection")
    args = parser.parse_args()

    print("Loading production prediction logs...")
    prod_df = load_prediction_logs(args.log_path)

    if len(prod_df) < 30:
        print(f"Warning: only {len(prod_df)} production samples found. "
              f"Drift results may be unreliable with small sample sizes.")

    print("Loading training data reference distribution...")
    training_lengths = load_training_text_lengths()
    production_lengths = prod_df["text_length"]

    results = check_drift(training_lengths, production_lengths, alpha=args.alpha)

    print("\n--- Drift Check Results ---")
    for key, value in results.items():
        print(f"{key}: {value}")

    if results["drift_detected"]:
        print("\n⚠️  Drift detected: production text length distribution significantly differs from training.")
    else:
        print("\n✅ No significant drift detected.")


if __name__ == "__main__":
    main()
