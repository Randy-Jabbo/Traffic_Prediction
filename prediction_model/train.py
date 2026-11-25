"""Train and save one I-8 traffic-regression model.

Run from the repository root:
    python prediction_model/train.py path/to/i8_west_2021.csv --model linear_regression
"""

import argparse
import json
from pathlib import Path

import joblib

try:
    from .evaluation import evaluate_regression
    from .models import MODEL_BUILDERS, build_model
    from .preprocess import FEATURE_COLUMNS, chronological_split, prepare_data
except ImportError:
    from evaluation import evaluate_regression
    from models import MODEL_BUILDERS, build_model
    from preprocess import FEATURE_COLUMNS, chronological_split, prepare_data


def train(csv_path, model_name="linear_regression", output_path=None):
    data = prepare_data(csv_path)
    train_df, test_df = chronological_split(data)

    model = build_model(model_name)
    model.fit(train_df[FEATURE_COLUMNS], train_df["flow"])
    metrics, _ = evaluate_regression(
        model, test_df[FEATURE_COLUMNS], test_df["flow"]
    )

    if output_path is None:
        output_path = f"saved_models/{model_name}.joblib"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_COLUMNS}, output_path)

    results = {
        "model": model_name,
        "total_usable_observations": len(data),
        "training_observations": len(train_df),
        "testing_observations": len(test_df),
        **metrics,
    }
    print(json.dumps(results, indent=2))
    print(f"Saved model to {Path(output_path).resolve()}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--model", choices=MODEL_BUILDERS, default="linear_regression")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    train(args.csv_path, args.model, args.output)