"""Presentation-ready evaluation: probability quality, surface results, and feature groups.

Run from the project root:
    python -m src.advanced_evaluation

This script trains only on the chronological training partition and evaluates
once on the held-out chronological test partition.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

if __package__:
    from .model import train_model, predict_model, predict_probabilities
    from .train import chronological_split
else:
    from model import train_model, predict_model, predict_probabilities
    from train import chronological_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _feature_group(name):
    name = name.lower()
    if "rank" in name:
        return "Player ranking"
    if "recent_win_rate" in name:
        return "Recent form"
    if "surface_win_rate" in name or name == "surface":
        return "Surface performance"
    if "head_to_head" in name:
        return "Head-to-head"
    if "age" in name or "height" in name:
        return "Player physical profile"
    if "tourney_level" in name or name == "round":
        return "Tournament context"
    return "Other"


def main():
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data" / "processed_matches.csv")
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    splits = chronological_split(data)
    train, test = splits["train"], splits["test"]
    model = train_model(train, train["target"])
    y_true = test["target"].astype(int).to_numpy()
    y_pred = np.asarray(predict_model(model, test))
    y_prob = np.asarray(predict_probabilities(model, test))

    overall = {
        "test_matches": len(test),
        "test_start": str(pd.to_datetime(test["tourney_date"]).min().date()),
        "test_end": str(pd.to_datetime(test["tourney_date"]).max().date()),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "brier_score": brier_score_loss(y_true, y_prob),
    }
    pd.DataFrame([overall]).to_csv(results_dir / "advanced_metrics.csv", index=False)

    # Surface breakdown uses the same already-held-out predictions.
    surface_rows = []
    surfaces = test["surface"].fillna("Unknown").astype(str).to_numpy()
    for surface in sorted(set(surfaces)):
        mask = surfaces == surface
        if mask.sum() == 0:
            continue
        row = {
            "surface": surface,
            "matches": int(mask.sum()),
            "accuracy": accuracy_score(y_true[mask], y_pred[mask]),
            "log_loss": log_loss(y_true[mask], y_prob[mask], labels=[0, 1]),
            "brier_score": brier_score_loss(y_true[mask], y_prob[mask]),
        }
        row["roc_auc"] = (
            roc_auc_score(y_true[mask], y_prob[mask])
            if np.unique(y_true[mask]).size == 2
            else np.nan
        )
        surface_rows.append(row)
    surface_df = pd.DataFrame(surface_rows)
    surface_df.to_csv(results_dir / "surface_performance.csv", index=False)

    # Aggregate built-in XGBoost gain importance into human-readable groups.
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    grouped = {}
    for name, importance in zip(names, importances):
        clean_name = name.split("__", 1)[-1]
        group = _feature_group(clean_name)
        grouped[group] = grouped.get(group, 0.0) + float(importance)
    group_df = (
        pd.DataFrame(
            [{"feature_group": k, "importance_share": v} for k, v in grouped.items()]
        )
        .sort_values("importance_share", ascending=False)
    )
    group_df.to_csv(results_dir / "feature_group_importance.csv", index=False)

    # Reliability diagram: do predicted probabilities match observed frequencies?
    fig, ax = plt.subplots(figsize=(7, 6))
    CalibrationDisplay.from_predictions(
        y_true, y_prob, n_bins=10, strategy="quantile", name="XGBoost", ax=ax
    )
    ax.set_title("Probability calibration · chronological test set")
    fig.tight_layout()
    fig.savefig(results_dir / "calibration_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(surface_df["surface"], surface_df["accuracy"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("XGBoost accuracy by court surface · test set")
    for i, value in enumerate(surface_df["accuracy"]):
        ax.text(i, value + 0.015, f"{value:.1%}", ha="center")
    fig.tight_layout()
    fig.savefig(results_dir / "surface_accuracy.png", dpi=180)
    plt.close(fig)

    print("Chronological test set:", overall["test_matches"], "matches")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss", "brier_score"):
        print(f"{key}: {overall[key]:.4f}")
    print("\nAccuracy by surface:")
    print(surface_df[["surface", "matches", "accuracy"]].to_string(index=False))
    print("\nFeature-group importance (model-based, not causal):")
    print(group_df.to_string(index=False))
    print("\nSaved advanced metrics, surface breakdown, feature-group importance, and plots to", results_dir)


if __name__ == "__main__":
    main()
