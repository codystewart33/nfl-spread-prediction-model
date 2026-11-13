
"""NFL Spread Lab — forward-tracked model results dashboard.

Presentation only. Reads the frozen data written by track_results.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Model Results | NFL Spread Lab",
    page_icon="📈",
    layout="wide",
)

HISTORY_PATH = Path("data/prediction_history.csv")

TEAM_NAMES = {
    "ARI": "Cardinals", "ATL": "Falcons",
    "BAL": "Ravens", "BUF": "Bills",
    "CAR": "Panthers", "CHI": "Bears",
    "CIN": "Bengals", "CLE": "Browns",
    "DAL": "Cowboys", "DEN": "Broncos",
    "DET": "Lions", "GB": "Packers",
    "HOU": "Texans", "IND": "Colts",
    "JAX": "Jaguars", "KC": "Chiefs",
    "LA": "Rams", "LAC": "Chargers",
    "LV": "Raiders", "MIA": "Dolphins",
    "MIN": "Vikings", "NE": "Patriots",
    "NO": "Saints", "NYG": "Giants",
    "NYJ": "Jets", "PHI": "Eagles",
    "PIT": "Steelers", "SEA": "Seahawks",
    "SF": "49ers", "TB": "Buccaneers",
    "TEN": "Titans", "WAS": "Commanders",
}

ESPN_CODES = {
    "LA": "lar",
    "WAS": "wsh",
    "WSH": "wsh",
    "JAC": "jax",
}


def team_logo(team):
    code = ESPN_CODES.get(
        str(team).upper(),
        str(team).lower(),
    )

    return (
        "https://a.espncdn.com/i/teamlogos/"
        f"nfl/500/{code}.png"
    )


def readable(value, fallback="—"):
    if value is None or pd.isna(value):
        return fallback

    return str(value)


def price_label(value):
    number = pd.to_numeric(
        value,
        errors="coerce",
    )

    if pd.isna(number):
        return "—"

    if number > 0:
        return f"{int(number):+d}"

    return str(int(number))


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
    return (
        int(frame["status"].eq("WIN").sum()),
        int(frame["status"].eq("LOSS").sum()),
        int(frame["status"].eq("PUSH").sum()),
    )


@st.cache_data(ttl=120)
def load_history():
    return pd.read_csv(HISTORY_PATH)


# ============================================================
# WEBSITE DESIGN
# ============================================================

# Extra top padding keeps the header below
# Streamlit's top toolbar.

st.html("""
<style>

.block-container,
[data-testid="stAppViewBlockContainer"] {
    max-width: 1250px;
    padding-top: 5.5rem !important;
    padding-bottom: 3rem;
}

.results-hero {
    background: linear-gradient(
        115deg,
        #101e35 0%,
        #173956 55%,
        #125e68 100%
    );

    border: 1px solid rgba(147,197,253,.20);
    border-radius: 18px;
    padding: 25px 30px;
    margin: 0 0 1.2rem;
    color: #f8fafc;
}

.results-eyebrow {
    color: #82e8de;
    font-size: .75rem;
    letter-spacing: .14em;
    font-weight: 750;
}

.results-title {
    font-size: clamp(1.8rem, 3vw, 2.3rem);
    font-weight: 800;
    line-height: 1.22;
    margin: 8px 0;
}

.results-subtitle {
    color: #cbd5e1;
    font-size: .95rem;
    margin: 0;
}

[data-testid="stMetric"] {
    border: 1px solid rgba(140,150,170,.20);
    background: rgba(120,140,160,.045);
    border-radius: 12px;
    padding: 12px 15px;
}

[data-testid="stMetricValue"] {
    font-size: 1.55rem;
}

@media (max-width: 640px) {
    .block-container,
    [data-testid="stAppViewBlockContainer"] {
        padding-top: 5rem !important;
    }

    .results-hero {
        padding: 21px 19px;
    }
}

</style>

<div class="results-hero">
    <div class="results-eyebrow">
        NFL · VERIFIED PREGAME SNAPSHOTS
    </div>

    <div class="results-title">
        Model Results
    </div>

    <p class="results-subtitle">
        Locked selections · Weekly performance ·
        Cumulative results
    </p>
</div>
""")


# ============================================================
# LOAD RESULTS
# ============================================================

if not HISTORY_PATH.exists():
    st.info(
        "No prediction history yet. "
        "Run the GitHub model tracker to create it."
    )
    st.stop()

try:
    history = load_history()

except (
    OSError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
) as exc:
    st.error(
        f"Could not load prediction history: {exc}"
    )
    st.stop()

if history.empty:
    st.info(
        "The history file is empty. "
        "The first pregame snapshots will appear here."
    )
    st.stop()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required = {
    "season",
    "week",
    "away_team",
    "home_team",
    "signal",
    "status",
    "kickoff_utc",
    "model_pick",
    "consensus_edge",
    "profit_units",
}

missing = required.difference(
    history.columns
)

if missing:
    st.error(
        "Missing history columns: "
        + ", ".join(sorted(missing))
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
    if col in history:
        history[col] = pd.to_numeric(
            history[col],
            errors="coerce",
        )

history["signal"] = (
    history["signal"].fillna("PASS")
)

history["status"] = (
    history["status"].fillna("PENDING")
)

history["kickoff_sort"] = pd.to_datetime(
    history["kickoff_utc"],
    utc=True,
    errors="coerce",
)


# ============================================================
# SEASON SELECTOR
# ============================================================

seasons = sorted(
    history["season"]
    .dropna()
    .astype(int)
    .unique()
    .tolist(),
    reverse=True,
)

if not seasons:
    st.info(
        "No valid season found in prediction history."
    )
    st.stop()

filter_col, desc_col = st.columns(
    [1, 3],
    vertical_alignment="bottom",
)

with filter_col:
    selected_season = st.selectbox(
        "Season",
        seasons,
    )

with desc_col:
    st.caption(
        "Only predictions saved before kickoff "
        "are counted here. Current sportsbook "
        "lines may be different."
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

wins, losses, pushes = record(settled)

pending = int(
    len(signals) - len(settled)
)

profit = float(
    settled["profit_units"].fillna(0).sum()
)

win_rate = (
    100 * wins / (wins + losses)
    if wins + losses
    else None
)

roi = (
    100 * profit / len(settled)
    if len(settled)
    else None
)


# ============================================================
# MAIN DASHBOARD METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Locked 4+ Signals",
    len(signals),
)

m2.metric(
    "Record (W–L–P)",
    f"{wins}–{losses}–{pushes}",
)

m3.metric(
    "Win Rate",
    f"{win_rate:.1f}%"
    if win_rate is not None
    else "—",
)

m4.metric(
    "Units / ROI",
    f"{profit:+.2f} units"
    if len(settled)
    else "—",
    help=(
        f"{roi:+.1f}% ROI"
        if roi is not None
        else "No graded signals yet"
    ),
)

roi_caption = (
    f"ROI: {roi:+.1f}% · "
    if roi is not None
    else ""
)

st.caption(
    f"{len(settled)} settled · "
    f"{pending} pending · "
    f"{roi_caption}"
    "One hypothetical unit staked per official "
    "4+ signal; pushes count in settled stakes."
)


# ============================================================
# NEW BALANCED PERFORMANCE SECTION
# ============================================================

st.subheader("Season Performance")

weekly_col, trend_col = st.columns(
    [1, 1.3],
    gap="medium",
)


# ============================================================
# LEFT SIDE: WEEKLY BREAKDOWN
# ============================================================

with weekly_col:

    with st.container(border=True):

        st.markdown("**Weekly Breakdown**")

        st.caption(
            "Official 4+ signals organized by week"
        )

        if signals.empty:

            st.write(
                "No official signals recorded yet."
            )

        else:

            rows = []

            for week, group in signals.groupby(
                "week",
                sort=True,
            ):

                done = group.loc[
                    group["status"].isin([
                        "WIN",
                        "LOSS",
                        "PUSH",
                    ])
                ]

                w, l, p = record(done)

                week_profit = float(
                    done["profit_units"]
                    .fillna(0)
                    .sum()
                )

                rows.append({
                    "Week": int(week),

                    "Record": (
                        f"{w}-{l}-{p}"
                        if len(done)
                        else "—"
                    ),

                    "Pending": (
                        len(group) - len(done)
                    ),

                    "Units": (
                        round(week_profit, 2)
                        if len(done)
                        else np.nan
                    ),
                })

            weekly_table = pd.DataFrame(
                rows
            ).sort_values(
                "Week",
                ascending=False,
            )

            st.dataframe(
                weekly_table,
                hide_index=True,
                use_container_width=True,
                height=min(
                    290,
                    40 + 36 * (len(weekly_table) + 1),
                ),
            )

            st.caption(
                f"{len(signals)} official signals "
                f"across {len(weekly_table)} "
                "tracked week(s)"
            )


# ============================================================
# RIGHT SIDE: CUMULATIVE PERFORMANCE
# ============================================================

with trend_col:

    with st.container(border=True):

        st.markdown(
            "**Cumulative Performance**"
        )

        if settled.empty:

            st.caption(
                "No games have been graded yet."
            )

            st.progress(
                0.0,
                text=(
                    f"0 of {len(signals)} "
                    "official signals settled"
                ),
            )

            st.write(
                "The profit chart will appear "
                "when a locked 4+ signal is graded."
            )

            st.caption(
                "Only frozen pregame picks "
                "count toward this chart."
            )

        else:

            st.caption(
                "Profit in hypothetical units, "
                "in kickoff order"
            )

            trend = settled.sort_values([
                "kickoff_sort",
                "away_team",
                "home_team",
            ]).copy()

            trend["Game"] = np.arange(
                1,
                len(trend) + 1,
            )

            trend["Cumulative units"] = (
                trend["profit_units"]
                .fillna(0)
                .cumsum()
            )

            st.line_chart(
                trend.set_index("Game")[
                    ["Cumulative units"]
                ],
                height=235,
            )

            st.caption(
                f"{len(settled)} of {len(signals)} "
                "official signals settled"
            )


# ============================================================
# LOCKED MATCHUPS
# ============================================================

st.divider()

st.subheader("🏈 Locked Matchups")

left, right = st.columns(
    [1.5, 1]
)

with left:

    display = st.radio(
        "Show",
        [
            "Official 4+ signals",
            "All locked games",
        ],
        horizontal=True,
    )

with right:

    weeks = sorted(
        season_games["week"]
        .dropna()
        .astype(int)
        .unique()
        .tolist(),
        reverse=True,
    )

    week_choice = st.selectbox(
        "Week",
        ["All weeks"] + weeks,
    )


# ============================================================
# FILTER MATCHUPS
# ============================================================

shown = (
    signals.copy()
    if display == "Official 4+ signals"
    else season_games.copy()
)

if week_choice != "All weeks":

    shown = shown.loc[
        shown["week"].eq(week_choice)
    ]

shown = shown.sort_values(
    "kickoff_sort",
    ascending=False,
)


# ============================================================
# MATCHUP CARDS
# ============================================================

if shown.empty:

    st.info(
        "No saved matchups match your filters."
    )

else:

    for _, game in shown.iterrows():

        away = str(game["away_team"])
        home = str(game["home_team"])
        status = str(game["status"])

        with st.container(border=True):

            a_col, h_col, result_col = st.columns(
                [1.6, 1.6, 1.2],
                vertical_alignment="center",
            )

            # AWAY TEAM

            with a_col:

                a_logo, a_name = st.columns(
                    [1, 3],
                    vertical_alignment="center",
                )

                with a_logo:

                    st.image(
                        team_logo(away),
                        width=52,
                    )

                with a_name:

                    st.markdown(
                        f"**{away} · "
                        f"{TEAM_NAMES.get(away, away)}**"
                    )

                    st.caption("AWAY")

            # HOME TEAM

            with h_col:

                h_logo, h_name = st.columns(
                    [1, 3],
                    vertical_alignment="center",
                )

                with h_logo:

                    st.image(
                        team_logo(home),
                        width=52,
                    )

                with h_name:

                    st.markdown(
                        f"**{home} · "
                        f"{TEAM_NAMES.get(home, home)}**"
                    )

                    st.caption("HOME")

            # GAME RESULT

            with result_col:

                status_label = {
                    "WIN": "🟢 WIN",
                    "LOSS": "🔴 LOSS",
                    "PUSH": "⚪ PUSH",
                }.get(
                    status,
                    "⏳ PENDING",
                )

                st.markdown(
                    f"**{status_label}**"
                )

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

            st.caption(
                kickoff_label(
                    game["kickoff_utc"]
                )
            )

            st.divider()

            # --------------------------------
            # PREDICTION DETAILS
            # --------------------------------

            p1, p2, p3 = st.columns(
                3,
                gap="medium",
            )

            with p1:

                st.caption(
                    "LOCKED MODEL PICK"
                )

                st.markdown(
                    f"**{readable(game.get('model_pick'))}**"
                )

                st.caption(
                    "Saved signal: "
                    f"{readable(game.get('signal'))}"
                )

            with p2:

                st.caption(
                    "CONSENSUS EDGE"
                )

                edge = pd.to_numeric(
                    game.get("consensus_edge"),
                    errors="coerce",
                )

                if pd.notna(edge):

                    st.markdown(
                        f"**{abs(edge):.2f} pts**"
                    )

                else:

                    st.markdown("**—**")

                st.caption(
                    "Market: "
                    f"{readable(game.get('market_line'))}"
                )

            with p3:

                st.caption(
                    "LOCKED BEST SPORTSBOOK LINE"
                )

                st.markdown(
                    "**"
                    + readable(
                        game.get("best_available_line")
                    )
                    + "**"
                )

                st.caption(
                    f"{readable(game.get('best_bookmaker'))}"
                    " · "
                    f"{price_label(game.get('best_price'))}"
                    " odds"
                )

            # --------------------------------
            # TRACKED RETURNS
            # --------------------------------

            if (
                game["signal"] == "4+ EDGE"
                and status in {
                    "WIN",
                    "LOSS",
                    "PUSH",
                }
            ):

                units = pd.to_numeric(
                    game.get("profit_units"),
                    errors="coerce",
                )

                if pd.notna(units):

                    st.caption(
                        "Official tracked return: "
                        f"{units:+.2f} units"
                    )


# ============================================================
# COMPLETE PREDICTION HISTORY
# ============================================================

with st.expander(
    "View Complete Prediction History Table"
):

    rename = {
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
            c
            for c in rename
            if c in season_games
        ]
    ].rename(
        columns=rename
    ).copy()

    for col in ["Edge", "Units"]:

        if col in table:

            table[col] = pd.to_numeric(
                table[col],
                errors="coerce",
            ).round(2)

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# NAVIGATION
# ============================================================

if Path("app.py").exists():

    st.page_link(
        "app.py",
        label="Back to Current Predictions",
        icon="🏈",
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "Forward-tracked pregame snapshots only. "
    "These are model signals, not confirmed bets. "
    "Current sportsbook lines can change. "
    "Historical backtests are separate "
    "from these results."
)
