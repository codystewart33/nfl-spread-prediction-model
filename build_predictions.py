from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import os

import nflreadpy as nfl
import numpy as np
import pandas as pd
import requests

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# SETTINGS
# ============================================================

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


# ============================================================
# NFL TEAM NAME MAPPING
# ============================================================

TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}


# ============================================================
# SEASON
# ============================================================

def nfl_season_for_today() -> int:
    """
    Return the NFL season year for today's date.
    """
    today = datetime.now()

    return (
        today.year
        if today.month >= 3
        else today.year - 1
    )


# ============================================================
# LOAD NFL DATA
# ============================================================

def load_data(current_season: int):

    seasons = list(
        range(
            START_SEASON,
            current_season + 1,
        )
    )

    schedules = nfl.load_schedules(
        seasons
    ).to_pandas()

    pbp = nfl.load_pbp(
        seasons
    )

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

    pbp["success"] = (
        pbp["epa"] > 0
    ).astype(int)

    return schedules, pbp


# ============================================================
# COMPLETED GAMES
# ============================================================

def build_completed_games(
    schedules: pd.DataFrame
) -> pd.DataFrame:

    games = schedules[
        (schedules["game_type"] == "REG")
        & schedules["home_score"].notna()
        & schedules["away_score"].notna()
    ].copy()

    sort_cols = [
        c
        for c in [
            "season",
            "week",
            "gameday",
            "game_id",
        ]
        if c in games.columns
    ]

    games = games.sort_values(
        sort_cols
    )

    games["home_margin"] = (
        games["home_score"]
        - games["away_score"]
    )

    return games


# ============================================================
# WEEKLY PASS / RUSH STATS
# ============================================================

def build_weekly_split_stats(
    pbp: pd.DataFrame
) -> pd.DataFrame:

    pass_pbp = pbp[
        pbp["play_type"] == "pass"
    ].copy()

    rush_pbp = pbp[
        pbp["play_type"] == "run"
    ].copy()

    # ---------------------------
    # PASS OFFENSE
    # ---------------------------

    pass_offense = (
        pass_pbp
        .groupby([
            "season",
            "week",
            "posteam",
        ])
        .agg(
            pass_epa=("epa", "mean"),
            pass_success=("success", "mean"),
        )
        .reset_index()
        .rename(
            columns={
                "posteam": "team"
            }
        )
    )

    # ---------------------------
    # RUSH OFFENSE
    # ---------------------------

    rush_offense = (
        rush_pbp
        .groupby([
            "season",
            "week",
            "posteam",
        ])
        .agg(
            rush_epa=("epa", "mean"),
            rush_success=("success", "mean"),
        )
        .reset_index()
        .rename(
            columns={
                "posteam": "team"
            }
        )
    )

    # ---------------------------
    # PASS DEFENSE
    # ---------------------------

    pass_defense = (
        pass_pbp
        .groupby([
            "season",
            "week",
            "defteam",
        ])
        .agg(
            pass_def_epa=("epa", "mean"),
            pass_def_success=("success", "mean"),
        )
        .reset_index()
        .rename(
            columns={
                "defteam": "team"
            }
        )
    )

    # ---------------------------
    # RUSH DEFENSE
    # ---------------------------

    rush_defense = (
        rush_pbp
        .groupby([
            "season",
            "week",
            "defteam",
        ])
        .agg(
            rush_def_epa=("epa", "mean"),
            rush_def_success=("success", "mean"),
        )
        .reset_index()
        .rename(
            columns={
                "defteam": "team"
            }
        )
    )

    # ---------------------------
    # MERGE
    # ---------------------------

    weekly = (
        pass_offense
        .merge(
            rush_offense,
            on=[
                "season",
                "week",
                "team",
            ],
            how="outer",
        )
        .merge(
            pass_defense,
            on=[
                "season",
                "week",
                "team",
            ],
            how="outer",
        )
        .merge(
            rush_defense,
            on=[
                "season",
                "week",
                "team",
            ],
            how="outer",
        )
        .sort_values([
            "team",
            "season",
            "week",
        ])
    )

    # ---------------------------
    # ORIGINAL LAST-5 MODEL
    # ---------------------------

    for metric in SPLIT_METRICS:

        weekly[
            f"last5_{metric}"
        ] = (
            weekly
            .groupby("team")[metric]
            .transform(
                lambda s:
                s.shift(1)
                .rolling(
                    window=5,
                    min_periods=3,
                )
                .mean()
            )
        )

    return weekly


# ============================================================
# MATCHUP DIFFERENCES
# ============================================================

def add_matchup_differences(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    # ---------------------------
    # OFFENSE
    # ---------------------------

    df["pass_epa_diff"] = (
        df["home_pass_epa"]
        - df["away_pass_epa"]
    )

    df["rush_epa_diff"] = (
        df["home_rush_epa"]
        - df["away_rush_epa"]
    )

    # ---------------------------
    # DEFENSE
    #
    # Lower EPA allowed is better,
    # so away - home means positive
    # favors the home team.
    # ---------------------------

    df["pass_def_diff"] = (
        df["away_pass_def_epa"]
        - df["home_pass_def_epa"]
    )

    df["rush_def_diff"] = (
        df["away_rush_def_epa"]
        - df["home_rush_def_epa"]
    )

    # ---------------------------
    # SUCCESS RATE
    # ---------------------------

    df["pass_success_diff"] = (
        df["home_pass_success"]
        - df["away_pass_success"]
    )

    df["rush_success_diff"] = (
        df["home_rush_success"]
        - df["away_rush_success"]
    )

    # ---------------------------
    # DEFENSIVE SUCCESS RATE
    # ---------------------------

    df["pass_def_success_diff"] = (
        df["away_pass_def_success"]
        - df["home_pass_def_success"]
    )

    df["rush_def_success_diff"] = (
        df["away_rush_def_success"]
        - df["home_rush_def_success"]
    )

    return df


# ============================================================
# BUILD TRAINING FRAME
# ============================================================

def build_training_frame(
    schedules: pd.DataFrame,
    weekly: pd.DataFrame,
) -> pd.DataFrame:

    games = build_completed_games(
        schedules
    )

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

    split_features = weekly[
        feature_cols
    ].copy()

    # ---------------------------
    # HOME
    # ---------------------------

    home = split_features.rename(
        columns={
            "team": "home_team",
            "last5_pass_epa":
                "home_pass_epa",
            "last5_pass_success":
                "home_pass_success",
            "last5_rush_epa":
                "home_rush_epa",
            "last5_rush_success":
                "home_rush_success",
            "last5_pass_def_epa":
                "home_pass_def_epa",
            "last5_pass_def_success":
                "home_pass_def_success",
            "last5_rush_def_epa":
                "home_rush_def_epa",
            "last5_rush_def_success":
                "home_rush_def_success",
        }
    )

    # ---------------------------
    # AWAY
    # ---------------------------

    away = split_features.rename(
        columns={
            "team": "away_team",
            "last5_pass_epa":
                "away_pass_epa",
            "last5_pass_success":
                "away_pass_success",
            "last5_rush_epa":
                "away_rush_epa",
            "last5_rush_success":
                "away_rush_success",
            "last5_pass_def_epa":
                "away_pass_def_epa",
            "last5_pass_def_success":
                "away_pass_def_success",
            "last5_rush_def_epa":
                "away_rush_def_epa",
            "last5_rush_def_success":
                "away_rush_def_success",
        }
    )

    df = games.merge(
        home,
        on=[
            "season",
            "week",
            "home_team",
        ],
        how="left",
    )

    df = df.merge(
        away,
        on=[
            "season",
            "week",
            "away_team",
        ],
        how="left",
    )

    df = add_matchup_differences(
        df
    )

    df = df.dropna(
        subset=FEATURES
        + ["home_margin"]
    ).copy()

    return df


# ============================================================
# DETERMINE CURRENT NFL WEEK
# ============================================================

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
            f"No regular-season schedule "
            f"found for {current_season}."
        )

    incomplete = season_sched[
        schedules.loc[
            season_sched.index,
            "home_score",
        ].isna()
        |
        schedules.loc[
            season_sched.index,
            "away_score",
        ].isna()
    ]

    if incomplete.empty:

        raise RuntimeError(
            f"No uncompleted regular-season "
            f"games remain for "
            f"{current_season}."
        )

    return int(
        incomplete["week"].min()
    )


# ============================================================
# CURRENT TEAM FEATURES
# ============================================================

def current_team_features(
    weekly: pd.DataFrame,
    current_season: int,
    target_week: int,
) -> pd.DataFrame:

    # Only use games before the
    # target week.
    #
    # This prevents Thursday results
    # from leaking into Sunday games.

    history = weekly[
        (weekly["season"] < current_season)
        |
        (
            (weekly["season"] == current_season)
            & (weekly["week"] < target_week)
        )
    ].copy()

    history = history.sort_values(
        [
            "team",
            "season",
            "week",
        ]
    )

    recent_five = (
        history
        .groupby(
            "team",
            group_keys=False,
        )
        .tail(5)
    )

    current = (
        recent_five
        .groupby("team")[
            SPLIT_METRICS
        ]
        .mean()
        .reset_index()
    )

    return current


# ============================================================
# BUILD UPCOMING GAME FRAME
# ============================================================

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
            |
            schedules["away_score"].isna()
        )
    ].copy()

    if upcoming.empty:

        raise RuntimeError(
            f"No upcoming games found "
            f"for {current_season} "
            f"Week {target_week}."
        )

    # ---------------------------
    # HOME CURRENT FEATURES
    # ---------------------------

    home = current_features.rename(
        columns={
            "team":
                "home_team",
            "pass_epa":
                "home_pass_epa",
            "pass_success":
                "home_pass_success",
            "rush_epa":
                "home_rush_epa",
            "rush_success":
                "home_rush_success",
            "pass_def_epa":
                "home_pass_def_epa",
            "pass_def_success":
                "home_pass_def_success",
            "rush_def_epa":
                "home_rush_def_epa",
            "rush_def_success":
                "home_rush_def_success",
        }
    )

    # ---------------------------
    # AWAY CURRENT FEATURES
    # ---------------------------

    away = current_features.rename(
        columns={
            "team":
                "away_team",
            "pass_epa":
                "away_pass_epa",
            "pass_success":
                "away_pass_success",
            "rush_epa":
                "away_rush_epa",
            "rush_success":
                "away_rush_success",
            "pass_def_epa":
                "away_pass_def_epa",
            "pass_def_success":
                "away_pass_def_success",
            "rush_def_epa":
                "away_rush_def_epa",
            "rush_def_success":
                "away_rush_def_success",
        }
    )

    upcoming = upcoming.merge(
        home,
        on="home_team",
        how="left",
    )

    upcoming = upcoming.merge(
        away,
        on="away_team",
        how="left",
    )

    upcoming = add_matchup_differences(
        upcoming
    )

    upcoming = upcoming.dropna(
        subset=FEATURES
    ).copy()

    return upcoming


# ============================================================
# FETCH LIVE SPORTSBOOK SPREADS
# ============================================================

def fetch_live_spreads(
    api_key: str
):

    url = (
        "https://api.the-odds-api.com/"
        "v4/sports/"
        "americanfootball_nfl/odds"
    )

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "spreads",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    response = requests.get(
        url,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    events = response.json()

    rows = []

    for event in events:

        home_name = event.get(
            "home_team"
        )

        away_name = event.get(
            "away_team"
        )

        home_team = (
            TEAM_NAME_TO_ABBR.get(
                home_name
            )
        )

        away_team = (
            TEAM_NAME_TO_ABBR.get(
                away_name
            )
        )

        if (
            home_team is None
            or away_team is None
        ):
            continue

        home_points = []

        for bookmaker in event.get(
            "bookmakers",
            [],
        ):

            for market in bookmaker.get(
                "markets",
                [],
            ):

                if (
                    market.get("key")
                    != "spreads"
                ):
                    continue

                for outcome in market.get(
                    "outcomes",
                    [],
                ):

                    if (
                        outcome.get("name")
                        == home_name
                        and outcome.get(
                            "point"
                        )
                        is not None
                    ):

                        home_points.append(
                            float(
                                outcome[
                                    "point"
                                ]
                            )
                        )

        if not home_points:
            continue

        # Example:
        #
        # Sportsbook:
        # GB -3
        #
        # Our model convention:
        # positive = home favored
        #
        # Therefore:
        # -3 sportsbook
        # becomes +3 model spread_line.

        consensus_home_point = float(
            np.median(
                home_points
            )
        )

        live_spread_line = (
            -consensus_home_point
        )

        rows.append({
            "home_team":
                home_team,
            "away_team":
                away_team,
            "live_spread_line":
                live_spread_line,
            "book_count":
                len(home_points),
            "commence_time":
                event.get(
                    "commence_time"
                ),
        })

    # Always create expected columns,
    # even if no odds were returned.

    odds_df = pd.DataFrame(
        rows,
        columns=[
            "home_team",
            "away_team",
            "live_spread_line",
            "book_count",
            "commence_time",
        ],
    )

    remaining = response.headers.get(
        "x-requests-remaining"
    )

    used = response.headers.get(
        "x-requests-used"
    )

    print(
        f"Odds API credits "
        f"remaining: {remaining}"
    )

    print(
        f"Odds API credits "
        f"used: {used}"
    )

    return odds_df, remaining


# ============================================================
# DISPLAY HELPERS
# ============================================================

def format_model_pick(
    row
) -> str:

    if pd.isna(
        row.get("spread_line")
    ):
        return "NO LINE"

    spread = float(
        row["spread_line"]
    )

    if row["edge"] > 0:

        team = row[
            "home_team"
        ]

        team_spread = (
            -spread
        )

    else:

        team = row[
            "away_team"
        ]

        team_spread = spread

    if np.isclose(
        team_spread,
        0,
    ):
        return f"{team} PK"

    if team_spread > 0:

        return (
            f"{team} "
            f"+{team_spread:g}"
        )

    return (
        f"{team} "
        f"{team_spread:g}"
    )


def format_market_line(
    row
) -> str:

    if pd.isna(
        row.get("spread_line")
    ):
        return "NO LINE"

    spread = float(
        row["spread_line"]
    )

    if np.isclose(
        spread,
        0,
    ):
        return "PK"

    # Positive spread_line
    # means HOME favored.

    if spread > 0:

        return (
            f"{row['home_team']} "
            f"-{spread:g}"
        )

    return (
        f"{row['away_team']} "
        f"{spread:g}"
    )


def format_fair_spread(
    row
) -> str:

    margin = float(
        row["model_margin"]
    )

    if np.isclose(
        margin,
        0,
        atol=0.05,
    ):
        return "PK"

    if margin > 0:

        return (
            f"{row['home_team']} "
            f"-{abs(margin):.1f}"
        )

    return (
        f"{row['away_team']} "
        f"-{abs(margin):.1f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    current_season = (
        nfl_season_for_today()
    )

    # ---------------------------
    # LOAD DATA
    # ---------------------------

    schedules, pbp = load_data(
        current_season
    )

    weekly = (
        build_weekly_split_stats(
            pbp
        )
    )

    training_frame = (
        build_training_frame(
            schedules,
            weekly,
        )
    )

    target_week = (
        determine_target_week(
            schedules,
            current_season,
        )
    )

    print(
        f"Current season: "
        f"{current_season}"
    )

    print(
        f"Target week: "
        f"{target_week}"
    )

    # ---------------------------
    # TRAINING DATA
    # ---------------------------

    production_train = (
        training_frame[
            (
                training_frame[
                    "season"
                ]
                < current_season
            )
            |
            (
                (
                    training_frame[
                        "season"
                    ]
                    == current_season
                )
                &
                (
                    training_frame[
                        "week"
                    ]
                    < target_week
                )
            )
        ]
        .copy()
    )

    if production_train.empty:

        raise RuntimeError(
            "Training dataset is empty."
        )

    # ---------------------------
    # MODEL
    # ---------------------------

    model = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "ridge",
            Ridge(
                alpha=10
            ),
        ),
    ])

    model.fit(
        production_train[
            FEATURES
        ],
        production_train[
            "home_margin"
        ],
    )

    # ---------------------------
    # CURRENT TEAM RATINGS
    # ---------------------------

    team_features = (
        current_team_features(
            weekly,
            current_season=
                current_season,
            target_week=
                target_week,
        )
    )

    # ---------------------------
    # UPCOMING GAMES
    # ---------------------------

    upcoming = (
        build_upcoming_frame(
            schedules,
            team_features,
            current_season=
                current_season,
            target_week=
                target_week,
        )
    )

    # ========================================================
    # CURRENT SPORTSBOOK SPREADS
    # ========================================================

    api_key = os.getenv(
        "THE_ODDS_API_KEY"
    )

    # Preserve nflverse line
    # as backup.

    upcoming[
        "nflverse_spread_line"
    ] = upcoming[
        "spread_line"
    ]

    odds_remaining = None

    if api_key:

        try:

            live_odds, odds_remaining = (
                fetch_live_spreads(
                    api_key
                )
            )

            if not live_odds.empty:

                upcoming = (
                    upcoming.merge(
                        live_odds,
                        on=[
                            "home_team",
                            "away_team",
                        ],
                        how="left",
                    )
                )

            else:

                upcoming[
                    "live_spread_line"
                ] = np.nan

                upcoming[
                    "book_count"
                ] = 0

            # -----------------------
            # LIVE ODDS FIRST
            # -----------------------

            upcoming[
                "spread_line"
            ] = (
                upcoming[
                    "live_spread_line"
                ]
                .combine_first(
                    upcoming[
                        "nflverse_spread_line"
                    ]
                )
            )

            upcoming[
                "line_source"
            ] = np.where(
                upcoming[
                    "live_spread_line"
                ].notna(),
                "Live consensus",
                "nflverse fallback",
            )

        except Exception as e:

            print(
                "Live odds unavailable:",
                e,
            )

            upcoming[
                "live_spread_line"
            ] = np.nan

            upcoming[
                "book_count"
            ] = 0

            upcoming[
                "line_source"
            ] = (
                "nflverse fallback"
            )

    else:

        print(
            "THE_ODDS_API_KEY "
            "not found. "
            "Using nflverse spreads."
        )

        upcoming[
            "live_spread_line"
        ] = np.nan

        upcoming[
            "book_count"
        ] = 0

        upcoming[
            "line_source"
        ] = (
            "nflverse fallback"
        )

    # ========================================================
    # MODEL PREDICTIONS
    # ========================================================

    upcoming[
        "model_margin"
    ] = model.predict(
        upcoming[
            FEATURES
        ]
    )

    upcoming[
        "edge"
    ] = np.where(
        upcoming[
            "spread_line"
        ].notna(),
        upcoming[
            "model_margin"
        ]
        - upcoming[
            "spread_line"
        ],
        np.nan,
    )

    upcoming[
        "abs_edge"
    ] = upcoming[
        "edge"
    ].abs()

    # ---------------------------
    # FORMATTED OUTPUT
    # ---------------------------

    upcoming[
        "model_pick"
    ] = upcoming.apply(
        format_model_pick,
        axis=1,
    )

    upcoming[
        "market_line"
    ] = upcoming.apply(
        format_market_line,
        axis=1,
    )

    upcoming[
        "model_fair_spread"
    ] = upcoming.apply(
        format_fair_spread,
        axis=1,
    )

    upcoming[
        "signal"
    ] = np.select(
        [
            upcoming[
                "spread_line"
            ].isna(),

            upcoming[
                "abs_edge"
            ] >= EDGE_THRESHOLD,
        ],
        [
            "NO LINE",
            "4+ EDGE",
        ],
        default="PASS",
    )

    # ---------------------------
    # TIMESTAMP
    # ---------------------------

    now = (
        datetime
        .now()
        .astimezone()
    )

    upcoming[
        "updated_at"
    ] = now.isoformat(
        timespec="seconds"
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    keep_cols = [
        c
        for c in [
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

            "line_source",
            "book_count",

            "updated_at",
        ]
        if c in upcoming.columns
    ]

    output = upcoming[
        keep_cols
    ].copy()

    output = (
        output.sort_values(
            "abs_edge",
            ascending=False,
            na_position="last",
        )
    )

    # ---------------------------
    # DATA DIRECTORY
    # ---------------------------

    data_dir = Path(
        "data"
    )

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------
    # CSV
    # ---------------------------

    output.to_csv(
        data_dir
        / "current_predictions.csv",
        index=False,
    )

    # ========================================================
    # MODEL METADATA
    # ========================================================

    metadata = {

        "season":
            current_season,

        "week":
            target_week,

        "edge_threshold":
            EDGE_THRESHOLD,

        "training_games":
            int(
                len(
                    production_train
                )
            ),

        "generated_at":
            now.isoformat(
                timespec="seconds"
            ),

        "odds_api_credits_remaining":
            odds_remaining,

        "model":
            (
                "StandardScaler + "
                "Ridge(alpha=10)"
            ),

        "features":
            FEATURES,

        "historical_walk_forward_2021_2025": {

            "bets":
                466,

            "record":
                "246-210-10",

            "win_rate_pct":
                53.95,

            "profit_units_at_minus_110":
                13.64,

            "roi_pct":
                2.93,
        },

        "note": (
            "The 2021-2025 statistics "
            "are historical walk-forward "
            "results from the development "
            "process. They are not a "
            "guarantee of future "
            "performance."
        ),
    }

    with open(
        data_dir
        / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    print()

    print(
        f"Built "
        f"{current_season} "
        f"Week {target_week} "
        f"board with "
        f"{len(output)} "
        f"upcoming games."
    )

    print()

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
                "line_source",
                "book_count",
            ]
        ]
        .round(2)
        .to_string(
            index=False
        )
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
