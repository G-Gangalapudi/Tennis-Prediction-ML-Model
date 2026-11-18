
from pathlib import Path
import pandas as pd


def load_atp_matches(data_dir="data"):

    data_path = Path(data_dir)

    # Only load main ATP singles matches
    files = sorted(
        data_path.glob("atp_matches_[0-9][0-9][0-9][0-9].csv")
    )

    if not files:
        raise FileNotFoundError("No ATP match files found!")

    # Combine all yearly CSV files
    matches = pd.concat(
        [pd.read_csv(file) for file in files],
        ignore_index=True
    )

    # Convert tournament date
    matches["tourney_date"] = pd.to_datetime(
        matches["tourney_date"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )

    matches = matches.dropna(subset=["tourney_date"])

    matches = matches.sort_values("tourney_date")

    return matches


def inspect_dataset(matches):

    print("Dataset shape:", matches.shape)

    print("\nColumns:")
    print(matches.columns.tolist())

    print("\nMissing values:")
    print(matches.isnull().sum().sort_values(ascending=False).head(15))

    print("\nDate range:")
    print(matches["tourney_date"].min())
    print(matches["tourney_date"].max())

    print("\nSample matches:")
    print(matches[
        ["tourney_date", "surface", "winner_name", "loser_name",
         "winner_rank", "loser_rank"]
    ].head())


if __name__ == "__main__":

    matches = load_atp_matches()
    inspect_dataset(matches)
