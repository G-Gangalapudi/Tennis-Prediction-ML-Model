"""Evaluate two pre-match baselines on the ATP singles dataset."""

from pathlib import Path
from random import Random

import pandas as pd

if __package__:
    from .data_loader import load_atp_matches
else:
    from data_loader import load_atp_matches


def evaluate_baselines(matches, seed=42):
    """Score both baselines on the same matches, with fair coin flips as needed."""
    rng = Random(seed)
    random_correct = 0
    ranked_correct = 0
    ranks = matches[["winner_rank", "loser_rank"]].apply(
        pd.to_numeric, errors="coerce"
    )

    for winner_rank, loser_rank in ranks.itertuples(index=False, name=None):
        # Randomize which side contains the recorded winner.
        winner_on_left = rng.choice((True, False))
        left_rank, right_rank = (
            (winner_rank, loser_rank) if winner_on_left else (loser_rank, winner_rank)
        )
        random_pick_left = rng.choice((True, False))
        random_correct += random_pick_left == winner_on_left

        if pd.isna(left_rank) or pd.isna(right_rank) or left_rank == right_rank:
            ranked_pick_left = rng.choice((True, False))
        else:
            ranked_pick_left = left_rank < right_rank
        ranked_correct += ranked_pick_left == winner_on_left

    total_matches = len(matches)
    return pd.DataFrame(
        [
            {
                "baseline": baseline,
                "accuracy": correct / total_matches if total_matches else float("nan"),
                "correct_predictions": correct,
                "total_matches": total_matches,
            }
            for baseline, correct in (
                ("random", random_correct),
                ("higher_ranked_player", ranked_correct),
            )
        ]
    )


def main():
    project_dir = Path(__file__).resolve().parents[1]
    matches = load_atp_matches(project_dir / "data")
    results = evaluate_baselines(matches)
    results_dir = project_dir / "results"
    results_dir.mkdir(exist_ok=True)
    results.to_csv(results_dir / "baseline_results.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
