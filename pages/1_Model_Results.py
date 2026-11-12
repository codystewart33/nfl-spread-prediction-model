
"""NFL Model Results page for Streamlit."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="NFL Model Results",
    page_icon="📈",
    layout="wide",
)

st.title("📈 NFL Model Results")

st.caption(
    "Locked pre-kickoff predictions, graded against "
    "final NFL scores. Prices are frozen at the first "
    "recorded pregame snapshot."
)


# ============================================================
# LOAD PREDICTION HISTORY
# ============================================================

path = Path(
    "data/prediction_history.csv"
)

if not path.exists():
    st.info(
        "No prediction history yet. "
        "Run Update NFL model in GitHub Actions "
        "after adding the tracker."
    )
    st.stop()


@st.cache_data(ttl=120)
def load_history():
    return pd.read_csv(path)


history = load_history()

if history.empty:
    st.info(
        "The history file exists, but no eligible "
        "pre-kickoff live sportsbook quotes "
        "have been locked yet."
    )
    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

history["profit_units"] = pd.to_numeric(
    history["profit_units"],
    errors="coerce",
)

history["best_price"] = pd.to_numeric(
    history["best_price"],
    errors="coerce",
)

history["season"] = pd.to_numeric(
    history["season"],
    errors="coerce",
)

history["week"] = pd.to_numeric(
    history["week"],
    errors="coerce",
)

history["status"] = history[
    "status"
].fillna("PENDING")

history["signal"] = history[
    "signal"
].fillna("PASS")


# ============================================================
# SELECT SEASON
# ============================================================

seasons = sorted(
    history["season"]
    .dropna()
    .unique()
    .astype(int)
    .tolist(),
    reverse=True,
)

selected_season = st.selectbox(
    "Season",
    seasons,
)

season_history = history[
    history["season"] == selected_season
].copy()

signals = season_history[
    season_history["signal"] == "4+ EDGE"
].copy()

settled = signals[
    signals["status"].isin([
        "WIN", "LOSS", "PUSH"
    ])
].copy()


# ============================================================
# PERFORMANCE METRICS
# ============================================================

wins = int(
    (settled["status"] == "WIN").sum()
)

losses = int(
    (settled["status"] == "LOSS").sum()
)

pushes = int(
    (settled["status"] == "PUSH").sum()
)

unit_profit = float(
    settled["profit_units"].sum()
)

win_rate = (
    100.0 * wins / (wins + losses)
    if wins + losses
    else None
)

roi = (
    100.0 * unit_profit / len(settled)
    if len(settled)
    else None
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Record (W-L-P)",
    f"{wins}-{losses}-{pushes}",
)

m2.metric(
    "Win Rate",
    f"{win_rate:.1f}%"
    if win_rate is not None
    else "—",
)

m3.metric(
    "Profit",
    f"{unit_profit:+.2f} units",
)

m4.metric(
    "ROI / 1-unit signal",
    f"{roi:+.1f}%"
    if roi is not None
    else "—",
)

st.caption(
    f"{len(signals)} frozen 4+ point signals · "
    f"{len(settled)} settled · "
    f"{len(signals) - len(settled)} pending. "
    "Win rate excludes pushes. "
    "ROI includes pushes in settled stakes. "
    "Only 4+ consensus signals count toward profit, "
    "using saved best-book spreads and prices."
)


# ============================================================
# CUMULATIVE PROFIT CHART
# ============================================================

st.subheader(
    "📈 Cumulative Units"
)

if settled.empty:
    st.info(
        "No official signals have been graded yet. "
        "The chart will appear once games finish."
    )

else:
    chart_data = settled.sort_values([
        "kickoff_utc",
        "game_key",
    ]).copy()

    chart_data["cumulative_units"] = (
        chart_data["profit_units"].cumsum()
    )

    chart_data["game_number"] = np.arange(
        1,
        len(chart_data) + 1,
    )

    st.line_chart(
        chart_data
        .set_index("game_number")[
            "cumulative_units"
        ]
    )


# ============================================================
# WEEKLY PERFORMANCE
# ============================================================

st.subheader(
    "📊 Weekly Results"
)

if not settled.empty:

    weekly = (
        settled.groupby(
            "week",
            as_index=False,
        )
        .agg(
            games=("game_key", "size"),
            wins=(
                "status",
                lambda s: int((s == "WIN").sum()),
            ),
            losses=(
                "status",
                lambda s: int((s == "LOSS").sum()),
            ),
            pushes=(
                "status",
                lambda s: int((s == "PUSH").sum()),
            ),
            units=("profit_units", "sum"),
        )
    )

    weekly["units"] = (
        weekly["units"].round(2)
    )

    weekly = weekly.rename(
        columns={
            "week": "Week",
            "games": "Settled signals",
            "wins": "Wins",
            "losses": "Losses",
            "pushes": "Pushes",
            "units": "Units",
        }
    )

    st.dataframe(
        weekly,
        hide_index=True,
        use_container_width=True,
    )

else:
    st.caption(
        "Weekly performance will be shown "
        "after the first completed games."
    )


# ============================================================
# FROZEN PREDICTION HISTORY
# ============================================================

st.subheader(
    "🗂️ Frozen Prediction History"
)

filter_choice = st.radio(
    "Show",
    [
        "4+ EDGE signals",
        "All pregame predictions",
    ],
    horizontal=True,
)

if filter_choice == "4+ EDGE signals":
    view = signals.copy()
else:
    view = season_history.copy()


columns = {
    "week": "Week",
    "gameday": "Date",
    "away_team": "Away",
    "home_team": "Home",
    "model_pick": "Consensus Pick",
    "market_line": "Consensus",
    "best_available_line": "Locked Best Line",
    "best_bookmaker": "Sportsbook",
    "best_price": "Price",
    "consensus_edge": "Edge",
    "signal": "Signal",
    "home_score": "Home Score",
    "away_score": "Away Score",
    "status": "Result",
    "profit_units": "Units",
    "prediction_generated_at":
        "Prediction Created (UTC)",
}

shown = [
    c
    for c in columns
    if c in view.columns
]

view = (
    view.sort_values(
        ["week", "kickoff_utc"],
        ascending=[False, False],
    )[shown]
    .rename(columns=columns)
)

for col in ("Edge", "Units"):

    if col in view.columns:
        view[col] = pd.to_numeric(
            view[col],
            errors="coerce",
        ).round(2)

st.dataframe(
    view,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# DISCLAIMER
# ============================================================

st.caption(
    "These are recorded model signals, not proof "
    "that bets were placed. Historical backtest "
    "results on the main page are separate from "
    "this newly collected forward record."
)
