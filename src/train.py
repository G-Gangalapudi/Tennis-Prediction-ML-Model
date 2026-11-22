
"""Chronologically train and evaluate the ATP match winner model."""

from pathlib import Path
from random import Random

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

if __package__:
    from .model import (
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        predict_model,
        predict_probabilities,
        prepare_features,
        train_model,
    )
else:
    from model import (
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        predict_model,
        predict_probabilities,
        prepare_features,
        train_model,
    )


FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
IDENTIFIER_COLUMNS = {
    "tourney_id",
    "tourney_name",
    "tourney_date",
    "match_num",
    "player1_id",
    "player2_id",
    "player1_name",
    "player2_name",
    "winner_id",
    "loser_id",
    "winner_name",
    "loser_name",
    "match_id",
}


def chronological_split(matches):
    """Split matches by unique tournament dates into 70/15/15 partitions."""
    required_columns = {"tourney_date", "target"}
    missing_columns = required_columns.difference(matches.columns)
    if missing_columns:
        raise ValueError(
            "Processed matches are missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if matches.empty:
        raise ValueError("Processed matches contain no rows.")

    ordered = matches.copy()
    parsed_dates = pd.to_datetime(ordered["tourney_date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("Processed matches contain missing or invalid dates.")
    ordered["tourney_date"] = parsed_dates.to_numpy()

    targets = pd.to_numeric(ordered["target"], errors="coerce")
    if targets.isna().any() or not targets.isin((0, 1)).all():
        raise ValueError("The target must contain only binary labels 0 and 1.")
    ordered["target"] = targets.astype(int)

    unique_dates = pd.Index(ordered["tourney_date"].unique()).sort_values()
    if len(unique_dates) < 3:
        raise ValueError("At least three unique dates are needed for splitting.")

    train_end = int(len(unique_dates) * 0.70)
    validation_end = int(len(unique_dates) * 0.85)
    if train_end == 0 or validation_end <= train_end or validation_end >= len(
        unique_dates
    ):
        raise ValueError("Not enough unique dates to create all three splits.")

    date_groups = {
        "train": unique_dates[:train_end],
        "validation": unique_dates[train_end:validation_end],
        "test": unique_dates[validation_end:],
    }
    splits = {
        name: ordered.loc[ordered["tourney_date"].isin(dates)]
        .sort_values("tourney_date", kind="mergesort")
        .copy()
        for name, dates in date_groups.items()
    }

    if any(split.empty for split in splits.values()):
        raise ValueError("Chronological splitting produced an empty partition.")
    if not (
        splits["train"]["tourney_date"].max()
        < splits["validation"]["tourney_date"].min()
        and splits["validation"]["tourney_date"].max()
        < splits["test"]["tourney_date"].min()
    ):
        raise ValueError("Chronological splits overlap or are out of order.")
    return splits


def _features_and_target(split):
    """Return only model-approved inputs and the binary target."""
    features = prepare_features(split)
    if "target" in features.columns:
        raise ValueError("Target leakage: target is present in model features.")
    forbidden = IDENTIFIER_COLUMNS.intersection(features.columns)
    if forbidden:
        raise ValueError(
            "Identifier columns cannot be model features: "
            + ", ".join(sorted(forbidden))
        )
    if list(features.columns) != FEATURE_COLUMNS:
        raise ValueError("Prepared features do not match the model feature list.")
    return features, split["target"].to_numpy()


def _date_range(split):
    return "{} to {}".format(
        split["tourney_date"].min().date(),
        split["tourney_date"].max().date(),
    )


def _evaluate_split(model, split, split_name):
    features, targets = _features_and_target(split)
    predictions = np.asarray(predict_model(model, features))
    probabilities = np.asarray(predict_probabilities(model, features))

    if predictions.shape != targets.shape or not np.isin(
        predictions, (0, 1)
    ).all():
        raise ValueError(
            "{} predictions must be binary labels with one result per match."
            .format(split_name.capitalize())
        )
    if probabilities.shape != targets.shape or not np.isfinite(
        probabilities
    ).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(
            "{} probabilities must be finite values between 0 and 1."
            .format(split_name.capitalize())
        )

    has_both_classes = np.unique(targets).size == 2
    return {
        "dataset_split": split_name,
        "num_matches": len(split),
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(
            targets, predictions, zero_division=0
        ),
        "recall": recall_score(targets, predictions, zero_division=0),
        "f1_score": f1_score(targets, predictions, zero_division=0),
        "roc_auc": (
            roc_auc_score(targets, probabilities)
            if has_both_classes
            else float("nan")
        ),
    }


def evaluate_higher_ranked_baseline(test_data, seed=42):
    """Evaluate the higher-ranked-player baseline on the test set."""
    rng = Random(seed)

    player1_ranks = pd.to_numeric(
        test_data["player1_rank"], errors="coerce"
    )
    player2_ranks = pd.to_numeric(
        test_data["player2_rank"], errors="coerce"
    )

    targets = test_data["target"].to_numpy()
    correct = 0

    for rank1, rank2, target in zip(
        player1_ranks, player2_ranks, targets
    ):
        if pd.isna(rank1) or pd.isna(rank2) or rank1 == rank2:
            prediction = rng.choice((0, 1))
        else:
            prediction = 1 if rank1 < rank2 else 0

        correct += prediction == target

    total = len(test_data)

    return {
        "baseline": "higher_ranked_player",
        "accuracy": correct / total if total else float("nan"),
        "correct_predictions": correct,
        "total_matches": total,
    }


def run_training_and_evaluation(data_path=None, results_path=None):
    """Train on the earliest 70% of dates and evaluate on later dates."""
    project_root = Path(__file__).resolve().parents[1]
    data_path = (
        Path(data_path)
        if data_path
        else project_root / "data" / "processed_matches.csv"
    )
    results_path = (
        Path(results_path)
        if results_path
        else project_root / "results" / "model_evaluation.csv"
    )

    matches = pd.read_csv(data_path)
    splits = chronological_split(matches)
    for name in ("train", "validation", "test"):
        split = splits[name]
        print(
            "{}: {} matches, dates {}.".format(
                name.capitalize(), len(split), _date_range(split)
            )
        )

    X_train, y_train = _features_and_target(splits["train"])
    model = train_model(X_train, y_train)

    result_rows = [
        _evaluate_split(model, splits["validation"], "validation"),
        _evaluate_split(model, splits["test"], "test"),
    ]
    results = pd.DataFrame(
        result_rows,
        columns=[
            "dataset_split",
            "num_matches",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
        ],
    )
    for row in result_rows:
        print(
            "{} metrics ({} matches, dates {}): accuracy={:.4f}, "
            "precision={:.4f}, recall={:.4f}, F1={:.4f}, ROC-AUC={}.".format(
                row["dataset_split"].capitalize(),
                row["num_matches"],
                _date_range(splits[row["dataset_split"]]),
                row["accuracy"],
                row["precision"],
                row["recall"],
                row["f1_score"],
                (
                    "{:.4f}".format(row["roc_auc"])
                    if np.isfinite(row["roc_auc"])
                    else "undefined (one target class)"
                ),
            )
        )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    print("Saved evaluation results to {}.".format(results_path))

    baseline_result = evaluate_higher_ranked_baseline(splits["test"])

    print("\nHigher-ranked baseline on test set:")
    print("Accuracy: {:.4f}".format(baseline_result["accuracy"]))
    print(
        "Correct predictions: {}".format(
            baseline_result["correct_predictions"]
        )
    )
    print("Total matches: {}".format(baseline_result["total_matches"]))

    comparison_path = (
        project_root / "results" / "test_baseline_comparison.csv"
    )
    pd.DataFrame([baseline_result]).to_csv(
        comparison_path, index=False
    )
    print("Saved baseline comparison to {}.".format(comparison_path))

    return model, splits, results


if __name__ == "__main__":
    run_training_and_evaluation()
