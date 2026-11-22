"""Create final evaluation plots for the chronological ATP test set."""

from pathlib import Path
from random import Random

import matplotlib
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
)

if __package__:
    from .model import predict_model, predict_probabilities, train_model
    from .train import chronological_split, evaluate_higher_ranked_baseline
else:
    from model import predict_model, predict_probabilities, train_model
    from train import chronological_split, evaluate_higher_ranked_baseline

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def save_plot(figure, results_dir, filename):
    figure.savefig(results_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_test_predictions(targets, predictions, probabilities, results_dir):
    figure, axis = plt.subplots(figsize=(7, 5))
    confusion = ConfusionMatrixDisplay.from_predictions(
        targets,
        predictions,
        labels=[0, 1],
        display_labels=["Player 2 wins", "Player 1 wins"],
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    for label, cell in zip(("TN", "FP", "FN", "TP"), confusion.text_.flat):
        cell.set_text(f"{label}\n{int(cell.get_text()):,}")
    axis.set_title("XGBoost confusion matrix · chronological test set")
    save_plot(figure, results_dir, "confusion_matrix.png")

    figure, axis = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(
        targets, probabilities, name="XGBoost", ax=axis
    )
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    axis.set_title("XGBoost ROC curve · chronological test set")
    axis.legend(loc="lower right")
    save_plot(figure, results_dir, "roc_curve.png")

    figure, axis = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(
        targets, probabilities, name="XGBoost", ax=axis
    )
    axis.axhline(
        targets.mean(), linestyle="--", color="gray", label="Positive class share"
    )
    axis.scatter(
        recall_score(targets, predictions, zero_division=0),
        precision_score(targets, predictions, zero_division=0),
        color="#ce6225",
        zorder=3,
        label="Model decision threshold",
    )
    axis.set_title("XGBoost precision–recall curve · chronological test set")
    axis.legend(loc="lower left")
    save_plot(figure, results_dir, "precision_recall_curve.png")


def plot_feature_importance(model, results_dir):
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importance = pd.Series(
        model.named_steps["classifier"].feature_importances_,
        index=[name.split("__", 1)[1] for name in feature_names],
    )
    top_features = importance.nlargest(15).sort_values()
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(top_features.index, top_features, color="#247ba0")
    axis.set_xlabel("Normalized XGBoost feature importance (gain)")
    axis.set_title(f"XGBoost feature importance · top {len(top_features)} inputs")
    save_plot(figure, results_dir, "feature_importance.png")


def plot_model_comparison(accuracies, test_count, results_dir):
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(
        list(accuracies),
        list(accuracies.values()),
        color=["#788896", "#477794", "#247ba0"],
    )
    axis.bar_label(
        bars, labels=[f"{score:.2%}" for score in accuracies.values()], padding=5
    )
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.set_ylabel("Accuracy")
    axis.set_title(f"Model comparison · same {test_count:,} chronological test matches")
    save_plot(figure, results_dir, "model_comparison.png")


def main():
    project_root = Path(__file__).resolve().parents[1]
    matches = pd.read_csv(project_root / "data" / "processed_matches.csv")
    splits = chronological_split(matches)
    train, test = splits["train"], splits["test"]
    model = train_model(train, train["target"])
    targets = test["target"]
    predictions = predict_model(model, test)
    probabilities = predict_probabilities(model, test)

    rng = Random(42)
    random_predictions = [rng.choice((0, 1)) for _ in range(len(test))]
    accuracies = {
        "Random baseline": accuracy_score(targets, random_predictions),
        "Higher-ranked player": evaluate_higher_ranked_baseline(test)["accuracy"],
        "XGBoost": accuracy_score(targets, predictions),
    }

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    plot_test_predictions(targets, predictions, probabilities, results_dir)
    plot_feature_importance(model, results_dir)
    plot_model_comparison(accuracies, len(test), results_dir)

    start, end = test["tourney_date"].agg(["min", "max"])
    print(
        f"Chronological test set ({start.date()} to {end.date()}): "
        f"{len(test):,} matches"
    )
    for name, accuracy in accuracies.items():
        print(f"{name}: {accuracy:.2%}")
    print(f"Saved five plots to {results_dir}")


if __name__ == "__main__":
    main()
