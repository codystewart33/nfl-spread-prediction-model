
"""NFL Spread Lab | Official Model Results.

Displays frozen pre-kickoff quotes saved by track_results.py.
Does not revise picks or calculate new predictions.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Model Results | NFL Spread Lab",
    page_icon="📈",
    layout="wide",
)

HISTORY_PATH = Path("data/prediction_history.csv")

ESPN_CODES = {
    "LA": "lar",
    "WAS": "wsh",
    "WSH": "wsh",
    "JAC": "jax",
}

TEAM_NAMES = {
    "ARI": "Cardinals",
    "ATL": "Falcons",
    "BAL": "Ravens",
    "BUF": "Bills",
    "CAR": "Panthers",
    "CHI": "Bears",
    "CIN": "Bengals",
    "CLE": "Browns",
    "DAL": "Cowboys",
    "DEN": "Broncos",
    "DET": "Lions",
    "GB": "Packers",
    "HOU": "Texans",
    "IND": "Colts",
    "JAX": "Jaguars",
    "KC": "Chiefs",
    "LA": "Rams",
    "LAC": "Chargers",
    "LV": "Raiders",
    "MIA": "Dolphins",
    "MIN": "Vikings",
    "NE": "Patriots",
    "NO": "Saints",
    "NYG": "Giants",
    "NYJ": "Jets",
    "PHI": "Eagles",
    "PIT": "Steelers",
    "SEA": "Seahawks",
    "SF": "49ers",
    "TB": "Buccaneers",
    "TEN": "Titans",
    "WAS": "Commanders",
}


def logo(team):
    code = ESPN_CODES.get(
        str(team).upper(),
        str(team).lower(),
    )
    return (
        "https://a.espncdn.com/i/teamlogos/nfl/"
        f"500/{code}.png"
    )


def price_label(value):
    if pd.isna(value):
        return "—"

    num = int(float(value))
    return f"+{num}" if num > 0 else str(num)


def kickoff_label(value):
    stamp = pd.to_datetime(
        value,
        utc=True,
        errors="coerce",
    )

    if pd.isna(stamp):
        return "Date unavailable"

    return (
        stamp.tz_convert("America/New_York")
        .strftime("%a, %b %d · %I:%M %p ET")
    )


def record(frame):
    wins = int(frame["status"].eq("WIN").sum())
    losses = int(frame["status"].eq("LOSS").sum())
    pushes = int(frame["status"].eq("PUSH").sum())

    return wins, losses, pushes


@st.cache_data(ttl=120)
def load_history():
    return pd.read_csv(HISTORY_PATH)


# ============================================================
# PAGE DESIGN
# ============================================================

st.html("""
<style>
.block-container {
    max-width: 1280px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}

.results-hero {
    background: linear-gradient(
        120deg,
        #101e35,
        #173956 55%,
        #125e68
    );
    border-radius: 18px;
    border: 1px solid #325b75;
    padding: 25px 28px;
    color: #f8fafc;
    margin-bottom: 16px;
}

.results-eyebrow {
    font-size: .78rem;
    letter-spacing: .15em;
    color: #82e8de;
    font-weight: 700;
}

.results-title {
    font-size: 2.1rem;
    line-height: 1.25;
    font-weight: 800;
    margin: 6px 0;
}

.results-sub {
    font-size: .96rem;
    color: #cbd5e1;
    margin: 0;
}

div[data-testid="stMetric"] {
    border: 1px solid rgba(140,150,170,.2);
    background: rgba(120,140,160,.045);
    border-radius: 12px;
    padding: 12px 16px;
}

[data-testid="stMetricValue"] {
    font-size: 1.55rem;
}
</style>

<div class="results-hero">
    <div class="results-eyebrow">
        NFL · OFFICIAL FORWARD TRACKING
    </div>

    <div class="results-title">
        Model Results
    </div>

    <p class="results-sub">
        Frozen pregame selections · verified game scores ·
        performance by week
    </p>
</div>
""")


# ============================================================
# LOAD RESULTS
# ============================================================

if not HISTORY_PATH.exists():
    st.info(
        "No saved results yet. The tracker creates "
        "this file after a successful model update."
    )
    st.stop()

try:
    history = load_history()

except (
    OSError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
) as error:
    st.error(f"Could not load prediction history: {error}")
    st.stop()


required = {
    "season",
    "week",
    "away_team",
    "home_team",
    "status",
    "signal",
    "kickoff_utc",
    "model_pick",
    "consensus_edge",
    "profit_units",
}

missing = required - set(history.columns)

if missing:
    st.error(
        "Missing result columns: "
        + ", ".join(sorted(missing))
    )
    st.stop()

if history.empty:
    st.info(
        "The tracker is set up, but no pregame "
        "predictions have been saved yet."
    )
    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

numeric_columns = [
    "season",
    "week",
    "consensus_edge",
    "profit_units",
    "best_price",
    "home_score",
    "away_score",
]

for col in numeric_columns:
    if col in history.columns:
        history[col] = pd.to_numeric(
            history[col],
            errors="coerce",
        )

history["status"] = (
    history["status"].fillna("PENDING")
)

history["signal"] = (
    history["signal"].fillna("PASS")
)

history["kickoff_sort"] = pd.to_datetime(
    history["kickoff_utc"],
    utc=True,
    errors="coerce",
)


# ============================================================
# SELECT SEASON
# ============================================================

seasons = sorted(
    history["season"]
    .dropna()
    .astype(int)
    .unique(),
    reverse=True,
)

if not seasons:
    st.warning("No valid seasons found in saved history.")
    st.stop()

left, right = st.columns([1, 3])

with left:
    selected_season = st.selectbox(
        "Season",
        seasons,
    )

with right:
    st.caption(
        "These results use the **first eligible "
        "pre-kickoff quote**. Later changes on the "
        "homepage never rewrite them."
    )


# ============================================================
# OFFICIAL SIGNALS
# ============================================================

season_games = history.loc[
    history["season"].eq(selected_season)
].copy()

signals = season_games.loc[
    season_games["signal"].eq("4+ EDGE")
].copy()

settled = signals.loc[
    signals["status"].isin([
        "WIN",
        "LOSS",
        "PUSH",
    ])
].copy()

w, l, p = record(settled)

profit = float(
    settled["profit_units"].fillna(0).sum()
)

win_pct = (
    100 * w / (w + l)
    if w + l
    else None
)

roi = (
    100 * profit / len(settled)
    if len(settled)
    else None
)


# ============================================================
# PERFORMANCE METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Official 4+ Signals",
    len(signals),
)

m2.metric(
    "Record (W–L–P)",
    f"{w}–{l}–{p}",
)

m3.metric(
    "Win Rate",
    f"{win_pct:.1f}%"
    if win_pct is not None
    else "—",
)

m4.metric(
    "Profit / ROI",
    f"{profit:+.2f} units"
    if len(settled)
    else "—",
    delta=(
        f"{roi:+.1f}% ROI"
        if roi is not None
        else None
    ),
    delta_color="off",
)

st.caption(
    f"{len(settled)} settled official signals · "
    f"{len(signals) - len(settled)} pending · "
    "Win rate excludes pushes. ROI counts one "
    "unit staked per settled signal, including pushes."
)


# ============================================================
# WEEKLY PERFORMANCE CARDS
# ============================================================

st.divider()

st.subheader("📊 Weekly Performance")

if signals.empty:
    st.info(
        "No locked 4+ point signals have been "
        "recorded for this season."
    )

else:
    weeks = sorted(
        signals["week"]
        .dropna()
        .astype(int)
        .unique(),
        reverse=True,
    )

    for start in range(0, len(weeks), 3):
        cols = st.columns(3)

        for col, week in zip(
            cols,
            weeks[start:start + 3],
        ):
            frame = signals.loc[
                signals["week"].eq(week)
            ]

            completed = frame.loc[
                frame["status"].isin([
                    "WIN",
                    "LOSS",
                    "PUSH",
                ])
            ]

            ww, ll, pp = record(completed)

            week_profit = (
                completed["profit_units"]
                .fillna(0)
                .sum()
            )

            with col:
                with st.container(border=True):
                    st.markdown(f"**WEEK {week}**")

                    st.metric(
                        "Record",
                        f"{ww}-{ll}-{pp}",
                    )

                    st.metric(
                        "Units",
                        f"{week_profit:+.2f}"
                        if len(completed)
                        else "—",
                    )

                    st.caption(
                        f"{len(completed)} settled · "
                        f"{len(frame) - len(completed)} "
                        "pending"
                    )


# ============================================================
# CUMULATIVE PROFIT CHART
# ============================================================

st.subheader("📈 Cumulative Profit")

if settled.empty:
    st.info(
        "The profit chart will appear after "
        "the first official 4+ signal finishes "
        "and is graded."
    )

else:
    chart_data = settled.sort_values([
        "kickoff_sort",
        "away_team",
        "home_team",
    ]).copy()

    chart_data["Game"] = np.arange(
        1,
        len(chart_data) + 1,
    )

    chart_data["Cumulative Units"] = (
        chart_data["profit_units"]
        .fillna(0)
        .cumsum()
    )

    st.line_chart(
        chart_data.set_index("Game")[
            ["Cumulative Units"]
        ]
    )


# ============================================================
# LOCKED MATCHUP CARDS
# ============================================================

st.divider()

st.subheader("🏈 Locked Matchups")

choice = st.radio(
    "Display",
    [
        "Official 4+ signals",
        "All locked games",
    ],
    horizontal=True,
)

shown = (
    signals
    if choice == "Official 4+ signals"
    else season_games
)

shown = shown.sort_values(
    "kickoff_sort",
    ascending=False,
)

if shown.empty:
    st.info(
        "No games match this selection."
    )

else:
    for _, game in shown.iterrows():

        away = str(game["away_team"])
        home = str(game["home_team"])
        status = str(game["status"])

        with st.container(border=True):

            (
                a_logo,
                a_team,
                h_logo,
                h_team,
                result,
            ) = st.columns(
                [0.75, 1.8, 0.75, 1.8, 1.35],
                gap="small",
                vertical_alignment="center",
            )

            with a_logo:
                st.image(
                    logo(away),
                    width=58,
                )

            with a_team:
                st.markdown(
                    f"**{away} · "
                    f"{TEAM_NAMES.get(away, away)}**"
                )
                st.caption("AWAY")

            with h_logo:
                st.image(
                    logo(home),
                    width=58,
                )

            with h_team:
                st.markdown(
                    f"**{home} · "
                    f"{TEAM_NAMES.get(home, home)}**"
                )
                st.caption(
                    kickoff_label(
                        game["kickoff_utc"]
                    )
                )

            with result:
                badge = {
                    "WIN": "🟢 WIN",
                    "LOSS": "🔴 LOSS",
                    "PUSH": "⚪ PUSH",
                }.get(
                    status,
                    "🕒 PENDING",
                )

                st.markdown(f"**{badge}**")

                if (
                    pd.notna(game.get("away_score"))
                    and pd.notna(game.get("home_score"))
                ):
                    st.caption(
                        f"Final: {away} "
                        f"{int(game['away_score'])}"
                        f" – {home} "
                        f"{int(game['home_score'])}"
                    )

            st.divider()

            c1, c2, c3 = st.columns(3)

            with c1:
                st.caption("LOCKED MODEL PICK")
                st.markdown(
                    f"**{game.get('model_pick', '—')}**"
                )
                st.caption(
                    f"Official signal: {game['signal']}"
                )

            with c2:
                st.caption("CONSENSUS EDGE")

                if pd.notna(
                    game["consensus_edge"]
                ):
                    edge = abs(
                        float(game["consensus_edge"])
                    )
                    st.markdown(
                        f"**{edge:.2f} points**"
                    )
                else:
                    st.markdown("—")

                st.caption(
                    f"Consensus: "
                    f"{game.get('market_line', '—')}"
                )

            with c3:
                st.caption(
                    "LOCKED BEST SPORTSBOOK LINE"
                )

                st.markdown(
                    f"**{game.get('best_available_line', '—')}**"
                )

                st.caption(
                    f"{game.get('best_bookmaker', '—')} "
                    f"· {price_label(game.get('best_price'))} "
                    "odds"
                )

            if (
                game["signal"] == "4+ EDGE"
                and status in ["WIN", "LOSS", "PUSH"]
            ):
                if pd.notna(
                    game["profit_units"]
                ):
                    st.caption(
                        "Graded return: "
                        f"{float(game['profit_units']):+.2f} "
                        "units"
                    )


# ============================================================
# COMPLETE HISTORY TABLE
# ============================================================

with st.expander(
    "Complete Frozen Prediction Table"
):

    columns = {
        "week": "Week",
        "gameday": "Date",
        "away_team": "Away",
        "home_team": "Home",
        "model_pick": "Locked Pick",
        "signal": "Signal",
        "market_line": "Consensus",
        "consensus_edge": "Edge",
        "best_available_line": "Best Line",
        "best_bookmaker": "Sportsbook",
        "best_price": "Odds",
        "status": "Result",
        "profit_units": "Units",
    }

    table = season_games[
        [
            key
            for key in columns
            if key in season_games.columns
        ]
    ].rename(columns=columns)

    for col in ["Edge", "Units"]:
        if col in table:
            table[col] = table[col].round(2)

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# NAVIGATION & DISCLAIMER
# ============================================================

if Path("app.py").exists():
    st.page_link(
        "app.py",
        label="Back to Current Predictions",
        icon="🏈",
    )

st.divider()

st.caption(
    "Forward tracking only: these are recorded "
    "pregame model signals, not evidence of bets "
    "placed. Historical backtests and current odds "
    "snapshots are separate. A model edge is not "
    "a calibrated probability."
)
