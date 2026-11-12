
"""Freeze pregame NFL model snapshots and grade prior signals from final scores.

Designed for the existing NFL spread dashboard. Runs AFTER build_predictions.py.
Does not refit the model, change picks, or query the paid odds feed.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import nflreadpy as nfl
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
PREDICTIONS_FILE = DATA_DIR / "current_predictions.csv"
HISTORY_FILE = DATA_DIR / "prediction_history.csv"
EASTERN = ZoneInfo("America/New_York")

HISTORY_COLUMNS = [
    "game_key", "season", "week", "gameday", "gametime",
    "away_team", "home_team", "kickoff_utc",
    "prediction_generated_at", "locked_at_utc",
    "market_line", "model_fair_spread", "model_margin",
    "consensus_edge", "model_pick", "signal", "line_source",
    "best_available_line", "best_bookmaker", "best_price",
    "best_line_edge", "pick_team", "pick_spread",
    "status", "home_score", "away_score", "profit_units",
    "graded_at_utc",
]


def parse_game_time(gameday, gametime):
    """nflverse gameday/gametime are Eastern clock times."""
    if pd.isna(gameday) or pd.isna(gametime):
        return None

    date_str = str(gameday).split("T")[0].split(" ")[0]
    time_str = str(gametime).strip()

    if not date_str or not time_str:
        return None

    try:
        naive = datetime.fromisoformat(
            f"{date_str}T{time_str}"
        )
        return naive.replace(
            tzinfo=EASTERN
        ).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def parse_timestamp(value):
    if value is None or pd.isna(value):
        return None

    parsed = pd.to_datetime(
        value, utc=True, errors="coerce"
    )

    if pd.isna(parsed):
        return None

    return parsed.to_pydatetime()


def parse_best_line(value, pick_team):
    """Read 'ARI +12.5', 'SF -3', or 'NYG PK'."""
    if pd.isna(value) or not isinstance(value, str):
        return None

    match = re.fullmatch(
        r"([A-Z]{2,3})\s+([+-]?\d+(?:\.\d+)?|PK)",
        value.strip(),
    )

    if match is None or match.group(1) != pick_team:
        return None

    return (
        0.0
        if match.group(2) == "PK"
        else float(match.group(2))
    )


def key_for(row):
    return (
        f"{int(row['season'])}-W{int(row['week']):02d}-"
        f"{str(row['away_team'])}-{str(row['home_team'])}"
    )


def american_win_profit(price):
    """Profit on one staked unit when a selection wins."""
    price = float(price)

    if price == 0 or not np.isfinite(price):
        raise ValueError("Invalid American odds")

    return (
        price / 100.0
        if price > 0
        else 100.0 / abs(price)
    )


def empty_history():
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def update_history(
    predictions,
    schedules,
    history=None,
    now=None,
):
    """Keep the first pregame quote; later runs only grade it."""

    if now is None:
        now = datetime.now(timezone.utc)

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    now = now.astimezone(timezone.utc)
    now_s = now.isoformat(timespec="seconds")

    if history is None or history.empty:
        history = empty_history()
    else:
        history = history.copy().reindex(
            columns=HISTORY_COLUMNS
        )

    # Build a lookup of NFL games.
    schedule_lookup = {}

    if not schedules.empty:
        reg = schedules.loc[
            schedules["game_type"].eq("REG")
        ].copy()

        for _, game in reg.iterrows():
            schedule_lookup[key_for(game)] = game

    existing = set(
        history["game_key"].dropna().astype(str)
    )

    new_rows = []

    # ========================================================
    # SAVE NEW PREGAME PREDICTIONS
    # ========================================================

    for _, row in predictions.iterrows():

        try:
            key = key_for(row)
        except (KeyError, TypeError, ValueError):
            continue

        # Never overwrite a saved prediction.
        if key in existing:
            continue

        game = schedule_lookup.get(key)

        if game is None:
            continue

        kickoff = parse_game_time(
            game.get("gameday"),
            game.get("gametime"),
        )

        generated = parse_timestamp(
            row.get("updated_at")
        )

        if kickoff is None or generated is None:
            continue

        # Both the prediction timestamp and
        # current run must be before kickoff.
        if not (
            generated <= now < kickoff
            and generated < kickoff
        ):
            continue

        # Only lock current sportsbook quotes.
        if row.get("line_source") != "Live consensus":
            continue

        pick = str(
            row.get("model_pick", "")
        ).split(" ")[0]

        if pick not in (
            str(row["home_team"]),
            str(row["away_team"]),
        ):
            continue

        spread = parse_best_line(
            row.get("best_available_line"),
            pick,
        )

        if spread is None:
            continue

        if pd.isna(row.get("best_bookmaker")):
            continue

        try:
            price = float(row.get("best_price"))
            american_win_profit(price)
        except (ValueError, TypeError):
            continue

        new_rows.append({
            "game_key": key,
            "season": int(row["season"]),
            "week": int(row["week"]),
            "gameday": str(game["gameday"]),
            "gametime": str(game["gametime"]),
            "away_team": row["away_team"],
            "home_team": row["home_team"],
            "kickoff_utc": kickoff.isoformat(
                timespec="seconds"
            ),
            "prediction_generated_at":
                generated.isoformat(timespec="seconds"),
            "locked_at_utc": now_s,
            "market_line": row.get("market_line"),
            "model_fair_spread":
                row.get("model_fair_spread"),
            "model_margin": row.get("model_margin"),
            "consensus_edge": row.get("edge"),
            "model_pick": row.get("model_pick"),
            "signal": row.get("signal"),
            "line_source": row.get("line_source"),
            "best_available_line":
                row.get("best_available_line"),
            "best_bookmaker":
                row.get("best_bookmaker"),
            "best_price": price,
            "best_line_edge":
                row.get("best_line_edge"),
            "pick_team": pick,
            "pick_spread": spread,
            "status": "PENDING",
            "home_score": np.nan,
            "away_score": np.nan,
            "profit_units": np.nan,
            "graded_at_utc": None,
        })

        existing.add(key)

    if new_rows:
        additions = pd.DataFrame(
            new_rows,
            columns=HISTORY_COLUMNS,
        )

        history = (
            additions
            if history.empty
            else pd.concat(
                [history, additions],
                ignore_index=True,
            )
        )

    # ========================================================
    # GRADE PREVIOUSLY SAVED GAMES
    # ========================================================

    for idx, row in history.iterrows():

        if str(row.get("status")) in (
            "WIN", "LOSS", "PUSH"
        ):
            continue

        game = schedule_lookup.get(
            str(row["game_key"])
        )

        if game is None:
            continue

        kickoff = parse_timestamp(
            row.get("kickoff_utc")
        )

        # Wait at least six hours after kickoff.
        # This helps avoid in-progress scores.
        if (
            kickoff is None
            or now < kickoff + timedelta(hours=6)
        ):
            continue

        home_score = pd.to_numeric(
            game.get("home_score"),
            errors="coerce",
        )

        away_score = pd.to_numeric(
            game.get("away_score"),
            errors="coerce",
        )

        if pd.isna(home_score) or pd.isna(away_score):
            continue

        pick = str(row["pick_team"])

        spread = pd.to_numeric(
            row.get("pick_spread"),
            errors="coerce",
        )

        if pd.isna(spread):
            continue

        # Determine the result using the
        # locked sportsbook spread.
        if pick == str(row["home_team"]):
            cover = (
                float(home_score)
                - float(away_score)
                + float(spread)
            )

        elif pick == str(row["away_team"]):
            cover = (
                float(away_score)
                - float(home_score)
                + float(spread)
            )

        else:
            continue

        if abs(cover) < 1e-8:
            result = "PUSH"
        elif cover > 0:
            result = "WIN"
        else:
            result = "LOSS"

        history.at[idx, "status"] = result
        history.at[idx, "home_score"] = int(home_score)
        history.at[idx, "away_score"] = int(away_score)
        history.at[idx, "graded_at_utc"] = now_s

        # Only official 4+ EDGE signals
        # contribute to hypothetical profit.
        if row.get("signal") == "4+ EDGE":

            if result == "WIN":
                profit = american_win_profit(
                    row["best_price"]
                )
            elif result == "LOSS":
                profit = -1.0
            else:
                profit = 0.0

            history.at[idx, "profit_units"] = profit

    if not history.empty:
        history = history.sort_values([
            "season",
            "week",
            "kickoff_utc",
            "game_key",
        ])

    return history.reindex(
        columns=HISTORY_COLUMNS
    ).reset_index(drop=True)


# ============================================================
# MAIN
# ============================================================

def main():

    if not PREDICTIONS_FILE.exists():
        raise SystemExit(
            "No predictions file. "
            "Run python build_predictions.py first."
        )

    predictions = pd.read_csv(
        PREDICTIONS_FILE
    )

    history = (
        pd.read_csv(HISTORY_FILE)
        if HISTORY_FILE.exists()
        else empty_history()
    )

    seasons = set()

    for frame in (predictions, history):

        if not frame.empty and "season" in frame:
            seasons.update(
                pd.to_numeric(
                    frame["season"],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .tolist()
            )

    if not seasons:
        raise SystemExit(
            "No season found to track."
        )

    schedules = nfl.load_schedules(
        sorted(seasons)
    ).to_pandas()

    updated = update_history(
        predictions,
        schedules,
        history,
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    updated.to_csv(
        HISTORY_FILE,
        index=False,
    )

    signals = updated[
        updated["signal"] == "4+ EDGE"
    ]

    settled = signals[
        signals["status"].isin([
            "WIN", "LOSS", "PUSH"
        ])
    ]

    print(
        f"History: {len(updated)} pregame snapshots, "
        f"{len(signals)} official signals, "
        f"{len(settled)} settled."
    )

    if len(settled):
        profit = pd.to_numeric(
            settled["profit_units"],
            errors="coerce",
        ).sum()

        print(
            f"Profit: {profit:+.2f} units"
        )

    print(
        f"Saved {HISTORY_FILE}"
    )


if __name__ == "__main__":
    main()
