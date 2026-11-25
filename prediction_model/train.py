import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

try:
    from .evaluation import evaluate_regression
    from .models import MODEL_BUILDERS, build_model
    from .preprocess import (
        FEATURE_COLUMNS,
        chronological_split,
        prepare_data,
    )
except ImportError:
    from evaluation import evaluate_regression
    from models import MODEL_BUILDERS, build_model
    from preprocess import (
        FEATURE_COLUMNS,
        chronological_split,
        prepare_data,
    )


def train(csv_path, model_name="linear_regression", output_path=None):
    data = prepare_data(csv_path)
    train_df, test_df = chronological_split(data)

    model = build_model(model_name)

    model.fit(
        train_df[FEATURE_COLUMNS],
        train_df["flow"],
    )

    metrics, predictions = evaluate_regression(
        model,
        test_df[FEATURE_COLUMNS],
        test_df["flow"],
    )

    if output_path is None:
        output_path = f"saved_models/{model_name}.joblib"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
        },
        output_path,
    )

    results_dir = Path("results") / model_name
    results_dir.mkdir(parents=True, exist_ok=True)

    prediction_output = pd.DataFrame({
        "timestamp": test_df["timestamp"].to_numpy(),
        "actual_flow": test_df["flow"].to_numpy(),
        "predicted_flow": predictions,
    })

    prediction_output["error"] = (
        prediction_output["actual_flow"]
        - prediction_output["predicted_flow"]
    )

    prediction_output.to_csv(
        results_dir / "predictions.csv",
        index=False,
    )

    results = {
        "model": model_name,
        "total_usable_observations": len(data),
        "training_observations": len(train_df),
        "testing_observations": len(test_df),
        **metrics,
    }

    with open(
        results_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    if model_name == "linear_regression":
        try:
            from .models.linear_regression import get_coefficients
        except ImportError:
            from models.linear_regression import get_coefficients

        coefficients = get_coefficients(
            model,
            FEATURE_COLUMNS,
        )

        coefficients.to_csv(
            results_dir / "coefficients.csv",
            index=False,
        )

    print(json.dumps(results, indent=2))
    print(f"Saved model to: {output_path}")
    print(f"Saved results to: {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("csv_path")

    parser.add_argument(
        "--model",
        choices=MODEL_BUILDERS,
        default="linear_regression",
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    train(
        args.csv_path,
        args.model,
        args.output,
    )