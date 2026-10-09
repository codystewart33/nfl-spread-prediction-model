from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import nflreadpy as nfl
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


START_SEASON = 2018
EDGE_THRESHOLD = 4.0

FEATURES = [
    "pass_epa_diff",
    "rush_epa_diff",
    "pass_def_diff",
    "rush_def_diff",
    "pass_success_diff",
    "rush_success_diff",
    "pass_def_success_diff",
    "rush_def_success_diff",
]

SPLIT_METRICS = [
    "pass_epa",
    "pass_success",
    "rush_epa",
    "rush_success",
    "pass_def_epa",
    "pass_def_success",
    "rush_def_epa",
    "rush_def_success",
]


def nfl_season_for_today() -> int:
    """NFL season year for today's date."""
    today = datetime.now()
    return today.year if today.month >= 3 else today.year - 1


def load_data(current_season: int):
    seasons = list(range(START_SEASON, current_season + 1))

    schedules = nfl.load_schedules(seasons).to_pandas()

    pbp = nfl.load_pbp(seasons)
    pbp = pbp.select([
        "season",
        "season_type",
        "week",
        "posteam",
        "defteam",
        "play_type",
        "epa",
    ]).to_pandas()

    pbp = pbp[
        (pbp["season_type"] == "REG")
        & (pbp["play_type"].isin(["pass", "run"]))
        & pbp["epa"].notna()
        & pbp["posteam"].notna()
        & pbp["defteam"].notna()
    ].copy()

    pbp["success"] = (pbp["epa"] > 0).astype(int)

    return schedules, pbp


def build_completed_games(schedules: pd.DataFrame) -> pd.DataFrame:
    games = schedules[
        (schedules["game_type"] == "REG")
        & schedules["home_score"].notna()
        & schedules["away_score"].notna()
    ].copy()

    sort_cols = [c for c in ["season", "week", "gameday", "game_id"] if c in games.columns]
    games = games.sort_values(sort_cols)

    games["home_margin"] = games["home_score"] - games["away_score"]
    return games


def build_weekly_split_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    pass_pbp = pbp[pbp["play_type"] == "pass"].copy()
    rush_pbp = pbp[pbp["play_type"] == "run"].copy()

    pass_offense = (
        pass_pbp.groupby(["season", "week", "posteam"])
        .agg(
            pass_epa=("epa", "mean"),
            pass_success=("success", "mean"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    rush_offense = (
        rush_pbp.groupby(["season", "week", "posteam"])
        .agg(
            rush_epa=("epa", "mean"),
            rush_success=("success", "mean"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    pass_defense = (
        pass_pbp.groupby(["season", "week", "defteam"])
        .agg(
            pass_def_epa=("epa", "mean"),
            pass_def_success=("success", "mean"),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    rush_defense = (
        rush_pbp.groupby(["season", "week", "defteam"])
        .agg(
            rush_def_epa=("epa", "mean"),
            rush_def_success=("success", "mean"),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    weekly = (
        pass_offense
        .merge(rush_offense, on=["season", "week", "team"], how="outer")
        .merge(pass_defense, on=["season", "week", "team"], how="outer")
        .merge(rush_defense, on=["season", "week", "team"], how="outer")
        .sort_values(["team", "season", "week"])
    )

    # Original model: equal-weight trailing five games, shifted one game
    # so the current game is never included in its own features.
    for metric in SPLIT_METRICS:
        weekly[f"last5_{metric}"] = (
            weekly.groupby("team")[metric]
            .transform(
                lambda s: s.shift(1)
                .rolling(window=5, min_periods=3)
                .mean()
            )
        )

    return weekly


def build_training_frame(
    schedules: pd.DataFrame,
    weekly: pd.DataFrame,
) -> pd.DataFrame:
    games = build_completed_games(schedules)

    feature_cols = [
        "season",
        "week",
        "team",
        "last5_pass_epa",
        "last5_pass_success",
        "last5_rush_epa",
        "last5_rush_success",
        "last5_pass_def_epa",
        "last5_pass_def_success",
        "last5_rush_def_epa",
        "last5_rush_def_success",
    ]
    split_features = weekly[feature_cols].copy()

    home = split_features.rename(columns={
        "team": "home_team",
        "last5_pass_epa": "home_pass_epa",
        "last5_pass_success": "home_pass_success",
        "last5_rush_epa": "home_rush_epa",
        "last5_rush_success": "home_rush_success",
        "last5_pass_def_epa": "home_pass_def_epa",
        "last5_pass_def_success": "home_pass_def_success",
        "last5_rush_def_epa": "home_rush_def_epa",
        "last5_rush_def_success": "home_rush_def_success",
    })

    away = split_features.rename(columns={
        "team": "away_team",
        "last5_pass_epa": "away_pass_epa",
        "last5_pass_success": "away_pass_success",
        "last5_rush_epa": "away_rush_epa",
        "last5_rush_success": "away_rush_success",
        "last5_pass_def_epa": "away_pass_def_epa",
        "last5_pass_def_success": "away_pass_def_success",
        "last5_rush_def_epa": "away_rush_def_epa",
        "last5_rush_def_success": "away_rush_def_success",
    })

    df = games.merge(
        home,
        on=["season", "week", "home_team"],
        how="left",
    ).merge(
        away,
        on=["season", "week", "away_team"],
        how="left",
    )

    df = add_matchup_differences(df)
    return df.dropna(subset=FEATURES + ["home_margin"]).copy()


def add_matchup_differences(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["pass_epa_diff"] = df["home_pass_epa"] - df["away_pass_epa"]
    df["rush_epa_diff"] = df["home_rush_epa"] - df["away_rush_epa"]

    # Lower defensive EPA allowed is better, hence away - home.
    df["pass_def_diff"] = df["away_pass_def_epa"] - df["home_pass_def_epa"]
    df["rush_def_diff"] = df["away_rush_def_epa"] - df["home_rush_def_epa"]

    df["pass_success_diff"] = (
        df["home_pass_success"] - df["away_pass_success"]
    )
    df["rush_success_diff"] = (
        df["home_rush_success"] - df["away_rush_success"]
    )

    # Lower defensive success rate allowed is better, hence away - home.
    df["pass_def_success_diff"] = (
        df["away_pass_def_success"] - df["home_pass_def_success"]
    )
    df["rush_def_success_diff"] = (
        df["away_rush_def_success"] - df["home_rush_def_success"]
    )

    return df


def determine_target_week(
    schedules: pd.DataFrame,
    current_season: int,
) -> int:
    season_sched = schedules[
        (schedules["season"] == current_season)
        & (schedules["game_type"] == "REG")
    ].copy()

    if season_sched.empty:
        raise RuntimeError(
            f"No regular-season schedule found for {current_season}."
        )

    # First week that still contains at least one uncompleted game.
    incomplete = season_sched[
        season_sched["home_score"].isna()
        | season_sched["away_score"].isna()
    ]

    if incomplete.empty:
        raise RuntimeError(
            f"No uncompleted regular-season games remain for {current_season}."
        )

    return int(incomplete["week"].min())


def current_team_features(
    weekly: pd.DataFrame,
    current_season: int,
    target_week: int,
) -> pd.DataFrame:
    # IMPORTANT: only use weeks strictly before the target week.
    # This prevents a Thursday game from leaking into Sunday's predictions.
    history = weekly[
        (weekly["season"] < current_season)
        | (
            (weekly["season"] == current_season)
            & (weekly["week"] < target_week)
        )
    ].copy()

    history = history.sort_values(["team", "season", "week"])

    recent_five = (
        history.groupby("team", group_keys=False)
        .tail(5)
    )

    return (
        recent_five
        .groupby("team")[SPLIT_METRICS]
        .mean()
        .reset_index()
    )


def build_upcoming_frame(
    schedules: pd.DataFrame,
    current_features: pd.DataFrame,
    current_season: int,
    target_week: int,
) -> pd.DataFrame:
    upcoming = schedules[
        (schedules["season"] == current_season)
        & (schedules["game_type"] == "REG")
        & (schedules["week"] == target_week)
        & (
            schedules["home_score"].isna()
            | schedules["away_score"].isna()
        )
    ].copy()

    if upcoming.empty:
        raise RuntimeError(
            f"No upcoming games found for {current_season} Week {target_week}."
        )

    home = current_features.rename(columns={
        "team": "home_team",
        "pass_epa": "home_pass_epa",
        "pass_success": "home_pass_success",
        "rush_epa": "home_rush_epa",
        "rush_success": "home_rush_success",
        "pass_def_epa": "home_pass_def_epa",
        "pass_def_success": "home_pass_def_success",
        "rush_def_epa": "home_rush_def_epa",
        "rush_def_success": "home_rush_def_success",
    })

    away = current_features.rename(columns={
        "team": "away_team",
        "pass_epa": "away_pass_epa",
        "pass_success": "away_pass_success",
        "rush_epa": "away_rush_epa",
        "rush_success": "away_rush_success",
        "pass_def_epa": "away_pass_def_epa",
        "pass_def_success": "away_pass_def_success",
        "rush_def_epa": "away_rush_def_epa",
        "rush_def_success": "away_rush_def_success",
    })

    upcoming = upcoming.merge(home, on="home_team", how="left")
    upcoming = upcoming.merge(away, on="away_team", how="left")

    upcoming = add_matchup_differences(upcoming)
    return upcoming.dropna(subset=FEATURES).copy()


def format_model_pick(row) -> str:
    if pd.isna(row.get("spread_line")):
        return "NO LINE"

    spread = float(row["spread_line"])

    if row["edge"] > 0:
        team = row["home_team"]
        team_spread = -spread
    else:
        team = row["away_team"]
        team_spread = spread

    if np.isclose(team_spread, 0):
        return f"{team} PK"
    if team_spread > 0:
        return f"{team} +{team_spread:g}"
    return f"{team} {team_spread:g}"


def format_market_line(row) -> str:
    if pd.isna(row.get("spread_line")):
        return "NO LINE"

    spread = float(row["spread_line"])

    # nflverse convention:
    # positive spread_line = home team favored
    # negative spread_line = away team favored
    if np.isclose(spread, 0):
        return "PK"
    if spread > 0:
        return f"{row['home_team']} -{spread:g}"
    return f"{row['away_team']} {spread:g}"


def format_fair_spread(row) -> str:
    margin = float(row["model_margin"])

    if np.isclose(margin, 0, atol=0.05):
        return "PK"
    if margin > 0:
        return f"{row['home_team']} -{abs(margin):.1f}"
    return f"{row['away_team']} -{abs(margin):.1f}"


def main():
    current_season = nfl_season_for_today()

    schedules, pbp = load_data(current_season)
    weekly = build_weekly_split_stats(pbp)
    training_frame = build_training_frame(schedules, weekly)
    target_week = determine_target_week(schedules, current_season)

    production_train = training_frame[
        (training_frame["season"] < current_season)
        | (
            (training_frame["season"] == current_season)
            & (training_frame["week"] < target_week)
        )
    ].copy()

    if production_train.empty:
        raise RuntimeError("Training dataset is empty.")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10)),
    ])

    model.fit(
        production_train[FEATURES],
        production_train["home_margin"],
    )

    team_features = current_team_features(
        weekly,
        current_season=current_season,
        target_week=target_week,
    )

    upcoming = build_upcoming_frame(
        schedules,
        team_features,
        current_season=current_season,
        target_week=target_week,
    )

    upcoming["model_margin"] = model.predict(upcoming[FEATURES])

    upcoming["edge"] = np.where(
        upcoming["spread_line"].notna(),
        upcoming["model_margin"] - upcoming["spread_line"],
        np.nan,
    )

    upcoming["abs_edge"] = upcoming["edge"].abs()
    upcoming["model_pick"] = upcoming.apply(format_model_pick, axis=1)
    upcoming["market_line"] = upcoming.apply(format_market_line, axis=1)
    upcoming["model_fair_spread"] = upcoming.apply(format_fair_spread, axis=1)

    upcoming["signal"] = np.select(
        [
            upcoming["spread_line"].isna(),
            upcoming["abs_edge"] >= EDGE_THRESHOLD,
        ],
        [
            "NO LINE",
            "4+ EDGE",
        ],
        default="PASS",
    )

    now = datetime.now().astimezone()
    upcoming["updated_at"] = now.isoformat(timespec="seconds")

    keep_cols = [
        c for c in [
            "season",
            "week",
            "gameday",
            "gametime",
            "away_team",
            "home_team",
            "market_line",
            "spread_line",
            "model_fair_spread",
            "model_margin",
            "edge",
            "abs_edge",
            "model_pick",
            "signal",
            "updated_at",
        ]
        if c in upcoming.columns
    ]

    output = upcoming[keep_cols].copy()
    output = output.sort_values(
        "abs_edge",
        ascending=False,
        na_position="last",
    )

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    output.to_csv(
        data_dir / "current_predictions.csv",
        index=False,
    )

    metadata = {
        "season": current_season,
        "week": target_week,
        "edge_threshold": EDGE_THRESHOLD,
        "training_games": int(len(production_train)),
        "generated_at": now.isoformat(timespec="seconds"),
        "model": "StandardScaler + Ridge(alpha=10)",
        "features": FEATURES,
        "historical_walk_forward_2021_2025": {
            "bets": 466,
            "record": "246-210-10",
            "win_rate_pct": 53.95,
            "profit_units_at_minus_110": 13.64,
            "roi_pct": 2.93,
        },
        "note": (
            "The 2021-2025 statistics are historical walk-forward results from "
            "the development process. They are not a guarantee of future performance."
        ),
    }

    with open(
        data_dir / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Built {current_season} Week {target_week} board "
        f"with {len(output)} upcoming games."
    )
    print(
        output[
            [
                "away_team",
                "home_team",
                "market_line",
                "model_fair_spread",
                "edge",
                "model_pick",
                "signal",
            ]
        ].round(2).to_string(index=False)
    )


if __name__ == "__main__":
    main()
