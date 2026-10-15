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
# TEAM NAME MAPPING
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
# WEEKLY SPLIT STATS
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

    # Equal-weight trailing 5.
    # shift(1) keeps the current game
    # out of its own features.

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

    df["pass_epa_diff"] = (
        df["home_pass_epa"]
        - df["away_pass_epa"]
    )

    df["rush_epa_diff"] = (
        df["home_rush_epa"]
        - df["away_rush_epa"]
    )

    # Lower defensive EPA allowed
    # is better.

    df["pass_def_diff"] = (
        df["away_pass_def_epa"]
        - df["home_pass_def_epa"]
    )

    df["rush_def_diff"] = (
        df["away_rush_def_epa"]
        - df["home_rush_def_epa"]
    )

    df["pass_success_diff"] = (
        df["home_pass_success"]
        - df["away_pass_success"]
    )

    df["rush_success_diff"] = (
        df["home_rush_success"]
        - df["away_rush_success"]
    )

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
# TRAINING FRAME
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

    return df.dropna(
        subset=
            FEATURES
            + ["home_margin"]
    ).copy()


# ============================================================
# TARGET WEEK
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
        season_sched["home_score"].isna()
        |
        season_sched["away_score"].isna()
    ].copy()

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

    # Only use games strictly before
    # the target week.

    history = weekly[
        (weekly["season"] < current_season)
        |
        (
            (weekly["season"] == current_season)
            & (weekly["week"] < target_week)
        )
    ].copy()

    history = history.sort_values([
        "team",
        "season",
        "week",
    ])

    recent_five = (
        history
        .groupby(
            "team",
            group_keys=False,
        )
        .tail(5)
    )

    return (
        recent_five
        .groupby("team")[
            SPLIT_METRICS
        ]
        .mean()
        .reset_index()
    )


# ============================================================
# UPCOMING GAMES
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

    upcoming = (
        add_matchup_differences(
            upcoming
        )
    )

    return upcoming.dropna(
        subset=FEATURES
    ).copy()


# ============================================================
# LIVE SPORTSBOOK ODDS
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

    consensus_rows = []
    book_rows = []

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

            book_key = bookmaker.get(
                "key"
            )

            book_title = bookmaker.get(
                "title"
            )

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

                    outcome_name = (
                        outcome.get(
                            "name"
                        )
                    )

                    point = outcome.get(
                        "point"
                    )

                    price = outcome.get(
                        "price"
                    )

                    team = (
                        TEAM_NAME_TO_ABBR.get(
                            outcome_name
                        )
                    )

                    if (
                        team is None
                        or point is None
                    ):
                        continue

                    point = float(
                        point
                    )

                    if price is not None:
                        price = int(
                            price
                        )

                    book_rows.append({
                        "home_team":
                            home_team,

                        "away_team":
                            away_team,

                        "team":
                            team,

                        "bookmaker_key":
                            book_key,

                        "bookmaker":
                            book_title,

                        "book_spread":
                            point,

                        "book_price":
                            price,
                    })

                    if (
                        outcome_name
                        == home_name
                    ):

                        home_points.append(
                            point
                        )

        if not home_points:
            continue

        # Sportsbook:
        # home -3
        #
        # Model convention:
        # +3 means home favored by 3.

        consensus_home_point = float(
            np.median(
                home_points
            )
        )

        live_spread_line = (
            -consensus_home_point
        )

        consensus_rows.append({
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

    odds_df = pd.DataFrame(
        consensus_rows,
        columns=[
            "home_team",
            "away_team",
            "live_spread_line",
            "book_count",
            "commence_time",
        ],
    )

    book_lines_df = pd.DataFrame(
        book_rows,
        columns=[
            "home_team",
            "away_team",
            "team",
            "bookmaker_key",
            "bookmaker",
            "book_spread",
            "book_price",
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

    return (
        odds_df,
        book_lines_df,
        remaining,
    )


# ============================================================
# FORMAT MODEL PICK
# ============================================================

def format_model_pick(
    row
) -> str:

    if pd.isna(
        row.get(
            "spread_line"
        )
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

        return (
            f"{team} PK"
        )

    if team_spread > 0:

        return (
            f"{team} "
            f"+{team_spread:g}"
        )

    return (
        f"{team} "
        f"{team_spread:g}"
    )


# ============================================================
# FORMAT MARKET LINE
# ============================================================

def format_market_line(
    row
) -> str:

    if pd.isna(
        row.get(
            "spread_line"
        )
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

    # Positive spread_line =
    # home team favored.

    if spread > 0:

        return (
            f"{row['home_team']} "
            f"-{spread:g}"
        )

    return (
        f"{row['away_team']} "
        f"{spread:g}"
    )


# ============================================================
# FORMAT MODEL FAIR SPREAD
# ============================================================

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
# BEST AVAILABLE SPORTSBOOK LINE
# ============================================================

def add_best_available_lines(
    upcoming: pd.DataFrame,
    book_lines: pd.DataFrame,
) -> pd.DataFrame:

    upcoming = upcoming.copy()

    upcoming[
        "best_bookmaker"
    ] = None

    upcoming[
        "best_spread"
    ] = np.nan

    upcoming[
        "best_price"
    ] = np.nan

    upcoming[
        "best_available_line"
    ] = None

    upcoming[
        "best_line_edge"
    ] = np.nan

    if book_lines.empty:
        return upcoming

    for idx, row in (
        upcoming.iterrows()
    ):

        if pd.isna(
            row["edge"]
        ):
            continue

        # Model side is determined from
        # the CONSENSUS market edge.

        if row["edge"] > 0:

            pick_team = row[
                "home_team"
            ]

            pick_is_home = True

        else:

            pick_team = row[
                "away_team"
            ]

            pick_is_home = False

        options = book_lines[
            (
                book_lines[
                    "home_team"
                ]
                == row[
                    "home_team"
                ]
            )
            &
            (
                book_lines[
                    "away_team"
                ]
                == row[
                    "away_team"
                ]
            )
            &
            (
                book_lines[
                    "team"
                ]
                == pick_team
            )
        ].copy()

        if options.empty:
            continue

        # Bigger number is always better
        # for the bettor:
        #
        # +7 > +6.5
        # -2.5 > -3

        best_spread = (
            options[
                "book_spread"
            ].max()
        )

        best_options = options[
            options[
                "book_spread"
            ]
            == best_spread
        ].copy()

        # If multiple books have the
        # same line, use the best juice.

        best_options = (
            best_options
            .sort_values(
                "book_price",
                ascending=False,
                na_position="last",
            )
        )

        best = (
            best_options.iloc[0]
        )

        bookmaker = best[
            "bookmaker"
        ]

        price = best[
            "book_price"
        ]

        if np.isclose(
            best_spread,
            0,
        ):

            formatted_line = (
                f"{pick_team} PK"
            )

        elif best_spread > 0:

            formatted_line = (
                f"{pick_team} "
                f"+{best_spread:g}"
            )

        else:

            formatted_line = (
                f"{pick_team} "
                f"{best_spread:g}"
            )

        # Edge versus actual best line.

        if pick_is_home:

            best_line_edge = (
                row[
                    "model_margin"
                ]
                + best_spread
            )

        else:

            best_line_edge = (
                best_spread
                - row[
                    "model_margin"
                ]
            )

        upcoming.at[
            idx,
            "best_bookmaker",
        ] = bookmaker

        upcoming.at[
            idx,
            "best_spread",
        ] = best_spread

        upcoming.at[
            idx,
            "best_price",
        ] = price

        upcoming.at[
            idx,
            "best_available_line",
        ] = formatted_line

        upcoming.at[
            idx,
            "best_line_edge",
        ] = best_line_edge

    return upcoming


# ============================================================
# MAIN
# ============================================================

def main():

    current_season = (
        nfl_season_for_today()
    )

    # --------------------------------------------------------
    # LOAD NFL DATA
    # --------------------------------------------------------

    schedules, pbp = (
        load_data(
            current_season
        )
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

    # --------------------------------------------------------
    # PRODUCTION TRAINING DATA
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CURRENT TEAM FEATURES
    # --------------------------------------------------------

    team_features = (
        current_team_features(
            weekly,
            current_season=
                current_season,
            target_week=
                target_week,
        )
    )

    # --------------------------------------------------------
    # UPCOMING GAMES
    # --------------------------------------------------------

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
    # LIVE SPORTSBOOK SPREADS
    # ========================================================

    api_key = os.getenv(
        "THE_ODDS_API_KEY"
    )

    upcoming[
        "nflverse_spread_line"
    ] = upcoming[
        "spread_line"
    ]

    odds_remaining = None

    book_lines = (
        pd.DataFrame()
    )

    if api_key:

        try:

            (
                live_odds,
                book_lines,
                odds_remaining,
            ) = fetch_live_spreads(
                api_key
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

            # Live consensus first.
            # nflverse is fallback.

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

            book_lines = (
                pd.DataFrame()
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

        book_lines = (
            pd.DataFrame()
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
    ] = (
        upcoming[
            "edge"
        ].abs()
    )

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

    # --------------------------------------------------------
    # 4+ CONSENSUS SIGNAL
    # --------------------------------------------------------

    upcoming[
        "signal"
    ] = np.select(
        [
            upcoming[
                "spread_line"
            ].isna(),

            upcoming[
                "abs_edge"
            ]
            >= EDGE_THRESHOLD,
        ],
        [
            "NO LINE",
            "4+ EDGE",
        ],
        default="PASS",
    )

    # ========================================================
    # BEST AVAILABLE SPORTSBOOK LINE
    # ========================================================

    upcoming = (
        add_best_available_lines(
            upcoming,
            book_lines,
        )
    )

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

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
    # SAVE CURRENT PREDICTIONS
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

            # Market
            "market_line",
            "spread_line",

            # Model
            "model_fair_spread",
            "model_margin",

            # Consensus edge
            "edge",
            "abs_edge",

            "model_pick",
            "signal",

            # Best available book
            "best_available_line",
            "best_bookmaker",
            "best_price",
            "best_line_edge",

            # Market metadata
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

    # --------------------------------------------------------
    # DATA DIRECTORY
    # --------------------------------------------------------

    data_dir = Path(
        "data"
    )

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    output.to_csv(
        data_dir
        / "current_predictions.csv",
        index=False,
    )

    # ========================================================
    # METADATA
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

    display_cols = [
        c
        for c in [
            "away_team",
            "home_team",
            "market_line",
            "model_fair_spread",
            "edge",
            "model_pick",
            "signal",
            "best_available_line",
            "best_bookmaker",
            "best_price",
            "best_line_edge",
            "line_source",
            "book_count",
        ]
        if c in output.columns
    ]

    print(
        output[
            display_cols
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