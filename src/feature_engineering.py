"""Build leakage-safe historical features for ATP singles matches."""

import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

if __package__:
    from .data_loader import load_atp_matches
else:
    from data_loader import load_atp_matches


IDENTIFIER_COLUMNS = (
    "tourney_id",
    "tourney_name",
    "tourney_date",
    "match_num",
    "surface",
    "tourney_level",
    "round",
)


def _normalize_player_id(value):
    if pd.isna(value):
        raise ValueError("Match is missing a player ID.")

    try:
        numeric_id = float(value)
    except (TypeError, ValueError):
        raise ValueError("Player IDs must be numeric.") from None

    if not math.isfinite(numeric_id) or not numeric_id.is_integer():
        raise ValueError("Player IDs must be finite integers.")
    return int(numeric_id)


def _numeric_value(value):
    if pd.isna(value):
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _difference(left, right):
    if math.isfinite(left) and math.isfinite(right):
        return left - right
    return float("nan")


def _text_value(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _parse_birth_date(value):
    if pd.isna(value):
        return pd.NaT
    try:
        numeric_value = float(value)
        if math.isfinite(numeric_value) and numeric_value.is_integer():
            value = str(int(numeric_value))
    except (TypeError, ValueError):
        pass
    return pd.to_datetime(str(value), format="%Y%m%d", errors="coerce")


def _build_player_lookup(players):
    if players is None or players.empty:
        return {}
    if "player_id" not in players.columns:
        raise ValueError("Player data must include a 'player_id' column.")

    lookup = {}
    for row in players.to_dict("records"):
        try:
            player_id = _normalize_player_id(row.get("player_id"))
        except ValueError:
            continue

        first_name = _text_value(row.get("name_first"))
        last_name = _text_value(row.get("name_last"))
        full_name = " ".join(
            name for name in (first_name, last_name) if name is not None
        )

        date_of_birth = _parse_birth_date(row.get("dob"))
        lookup.setdefault(
            player_id,
            {
                "name": full_name or None,
                "height": _numeric_value(row.get("height")),
                "date_of_birth": date_of_birth,
            },
        )
    return lookup


def _recent_win_rate(history, window):
    if history is None:
        return float("nan")

    # A window can cut through a date batch whose matches have no known order.
    # Use that batch's average rather than arbitrarily sorting its outcomes.
    remaining = window
    observed = 0
    expected_wins = 0.0
    for wins, match_count in reversed(history["dated_results"]):
        if remaining == 0:
            break
        included = min(remaining, match_count)
        expected_wins += included * wins / match_count
        observed += included
        remaining -= included

    return expected_wins / observed if observed else float("nan")


def _surface_win_rate(history, surface):
    if history is None or surface is None:
        return float("nan")
    wins, match_count = history["surface_results"].get(surface, (0, 0))
    return wins / match_count if match_count else float("nan")


def _age_at_date(value, date_of_birth, match_date):
    age = _numeric_value(value)
    if math.isfinite(age):
        return age
    if date_of_birth is None or pd.isna(date_of_birth):
        return float("nan")
    return (match_date - date_of_birth).days / 365.25


def build_historical_features(matches, players=None):
    """Return one Player 1 vs Player 2 feature row for each valid dated match.

    Player 1 is the competitor with the lower numeric ATP ID. Historical
    outcomes are added only after every match on a tournament date is
    featurized, so results from the current date cannot affect its features.
    """
    required_columns = {"tourney_date", "winner_id", "loser_id"}
    missing_columns = required_columns.difference(matches.columns)
    if missing_columns:
        raise ValueError(
            "Match data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    source_columns = (
        "winner_id",
        "winner_name",
        "winner_rank",
        "winner_age",
        "winner_ht",
        "loser_id",
        "loser_name",
        "loser_rank",
        "loser_age",
        "loser_ht",
    )
    columns = [
        column
        for column in IDENTIFIER_COLUMNS + source_columns
        if column in matches.columns
    ]
    working = matches.loc[:, columns].copy()

    if pd.api.types.is_datetime64_any_dtype(working["tourney_date"]):
        dates = pd.to_datetime(working["tourney_date"], errors="coerce")
    else:
        dates = pd.to_datetime(
            working["tourney_date"].astype(str),
            format="%Y%m%d",
            errors="coerce",
        )

    working["_feature_date"] = dates
    working["_source_order"] = range(len(working))
    working = working.loc[working["_feature_date"].notna()].copy()

    for column in ("winner_id", "loser_id"):
        normalized_ids = []
        for row_number, value in enumerate(working[column].tolist()):
            try:
                normalized_ids.append(_normalize_player_id(value))
            except ValueError as error:
                raise ValueError(
                    "{} at source row {}: {}".format(column, row_number, error)
                ) from error
        working["_{}".format(column)] = normalized_ids

    # A same-ID placeholder row cannot represent two distinct competitors.
    working = working.loc[working["_winner_id"] != working["_loser_id"]].copy()
    working = working.sort_values(
        ["_feature_date", "_source_order"], kind="mergesort"
    )
    positions = {column: index for index, column in enumerate(working.columns)}
    player_lookup = _build_player_lookup(players)

    player_history = {}
    head_to_head = defaultdict(lambda: [0, 0])
    output_rows = []

    for match_date, day_matches in working.groupby("_feature_date", sort=True):
        date_player_results = defaultdict(lambda: [0, 0])
        date_surface_results = defaultdict(lambda: [0, 0])
        date_head_to_head = defaultdict(lambda: [0, 0])
        date_outcomes = []

        for row in day_matches.itertuples(index=False, name=None):
            winner_id = row[positions["_winner_id"]]
            loser_id = row[positions["_loser_id"]]
            player1_id, player2_id = sorted((winner_id, loser_id))
            target = int(winner_id == player1_id)

            winner_name = _text_value(
                row[positions["winner_name"]]
                if "winner_name" in positions
                else None
            )
            loser_name = _text_value(
                row[positions["loser_name"]]
                if "loser_name" in positions
                else None
            )
            winner_metadata = player_lookup.get(winner_id, {})
            loser_metadata = player_lookup.get(loser_id, {})

            winner_age = _age_at_date(
                row[positions["winner_age"]]
                if "winner_age" in positions
                else None,
                winner_metadata.get("date_of_birth"),
                match_date,
            )
            loser_age = _age_at_date(
                row[positions["loser_age"]]
                if "loser_age" in positions
                else None,
                loser_metadata.get("date_of_birth"),
                match_date,
            )
            winner_height = _numeric_value(
                row[positions["winner_ht"]]
                if "winner_ht" in positions
                else None
            )
            loser_height = _numeric_value(
                row[positions["loser_ht"]]
                if "loser_ht" in positions
                else None
            )
            if not math.isfinite(winner_height):
                winner_height = winner_metadata.get("height", float("nan"))
            if not math.isfinite(loser_height):
                loser_height = loser_metadata.get("height", float("nan"))

            winner_rank = _numeric_value(
                row[positions["winner_rank"]]
                if "winner_rank" in positions
                else None
            )
            loser_rank = _numeric_value(
                row[positions["loser_rank"]]
                if "loser_rank" in positions
                else None
            )
            surface_value = (
                _text_value(row[positions["surface"]])
                if "surface" in positions
                else None
            )
            surface = surface_value.lower() if surface_value else None

            player1_history = player_history.get(player1_id)
            player2_history = player_history.get(player2_id)
            if player1_id == winner_id:
                player1_name, player2_name = winner_name, loser_name
                player1_rank, player2_rank = winner_rank, loser_rank
                player1_age, player2_age = winner_age, loser_age
                player1_height, player2_height = winner_height, loser_height
            else:
                player1_name, player2_name = loser_name, winner_name
                player1_rank, player2_rank = loser_rank, winner_rank
                player1_age, player2_age = loser_age, winner_age
                player1_height, player2_height = loser_height, winner_height

            if player1_name is None:
                player1_name = player_lookup.get(player1_id, {}).get("name")
            if player2_name is None:
                player2_name = player_lookup.get(player2_id, {}).get("name")

            pair_wins = head_to_head[(player1_id, player2_id)]
            feature_row = {
                column: (
                    match_date
                    if column == "tourney_date"
                    else row[positions[column]]
                )
                for column in IDENTIFIER_COLUMNS
                if column in positions
            }
            feature_row.update(
                {
                    "player1_id": player1_id,
                    "player2_id": player2_id,
                    "player1_name": player1_name,
                    "player2_name": player2_name,
                    "player1_rank": player1_rank,
                    "player2_rank": player2_rank,
                    "ranking_diff": _difference(player1_rank, player2_rank),
                    "player1_recent_win_rate_5": _recent_win_rate(
                        player1_history, 5
                    ),
                    "player2_recent_win_rate_5": _recent_win_rate(
                        player2_history, 5
                    ),
                    "player1_recent_win_rate_10": _recent_win_rate(
                        player1_history, 10
                    ),
                    "player2_recent_win_rate_10": _recent_win_rate(
                        player2_history, 10
                    ),
                    "player1_surface_win_rate": _surface_win_rate(
                        player1_history, surface
                    ),
                    "player2_surface_win_rate": _surface_win_rate(
                        player2_history, surface
                    ),
                    "head_to_head": pair_wins[0] - pair_wins[1],
                    "player1_age": player1_age,
                    "player2_age": player2_age,
                    "age_diff": _difference(player1_age, player2_age),
                    "player1_height": player1_height,
                    "player2_height": player2_height,
                    "height_diff": _difference(
                        player1_height, player2_height
                    ),
                    "target": target,
                }
            )
            output_rows.append(feature_row)
            date_outcomes.append((winner_id, loser_id, surface))

        # Add the complete date batch only after its features are calculated.
        for winner_id, loser_id, surface in date_outcomes:
            date_player_results[winner_id][0] += 1
            date_player_results[winner_id][1] += 1
            date_player_results[loser_id][1] += 1
            if surface is not None:
                date_surface_results[(winner_id, surface)][0] += 1
                date_surface_results[(winner_id, surface)][1] += 1
                date_surface_results[(loser_id, surface)][1] += 1

            pair = tuple(sorted((winner_id, loser_id)))
            date_head_to_head[pair][0 if winner_id == pair[0] else 1] += 1

        for player_id, (wins, match_count) in date_player_results.items():
            history = player_history.setdefault(
                player_id,
                {
                    "dated_results": [],
                    "surface_results": defaultdict(lambda: [0, 0]),
                },
            )
            history["dated_results"].append((wins, match_count))

        for (player_id, surface), (wins, match_count) in (
            date_surface_results.items()
        ):
            surface_history = player_history.setdefault(
                player_id,
                {
                    "dated_results": [],
                    "surface_results": defaultdict(lambda: [0, 0]),
                },
            )["surface_results"][surface]
            surface_history[0] += wins
            surface_history[1] += match_count

        for pair, (player1_wins, player2_wins) in date_head_to_head.items():
            head_to_head[pair][0] += player1_wins
            head_to_head[pair][1] += player2_wins

    leading_columns = [
        column for column in IDENTIFIER_COLUMNS if column in matches.columns
    ]
    feature_columns = [
        "player1_id",
        "player2_id",
        "player1_name",
        "player2_name",
        "player1_rank",
        "player2_rank",
        "ranking_diff",
        "player1_recent_win_rate_5",
        "player2_recent_win_rate_5",
        "player1_recent_win_rate_10",
        "player2_recent_win_rate_10",
        "player1_surface_win_rate",
        "player2_surface_win_rate",
        "head_to_head",
        "player1_age",
        "player2_age",
        "age_diff",
        "player1_height",
        "player2_height",
        "height_diff",
        "target",
    ]
    return pd.DataFrame(output_rows, columns=leading_columns + feature_columns)


def generate_processed_matches(data_dir="data", output_path=None):
    """Load project data, engineer features, and write the processed CSV."""
    data_path = Path(data_dir)
    matches = load_atp_matches(data_path)
    player_path = data_path / "atp_players.csv"
    players = (
        pd.read_csv(
            player_path,
            usecols=["player_id", "name_first", "name_last", "dob", "height"],
            dtype={"dob": "string"},
        )
        if player_path.is_file()
        else None
    )

    processed = build_historical_features(matches, players)
    output_path = (
        Path(output_path)
        if output_path
        else data_path / "processed_matches.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    return processed


def main():
    project_dir = Path(__file__).resolve().parents[1]
    processed = generate_processed_matches(project_dir / "data")
    print("Wrote {} processed matches.".format(len(processed)))


if __name__ == "__main__":
    main()
