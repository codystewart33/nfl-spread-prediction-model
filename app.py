
"""NFL Spread Lab dashboard with current vs. locked predictions.

Reads generated predictions and saved prediction history.
Does not retrain the model or change official tracked picks.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="NFL Spread Lab",
    page_icon="🏈",
    layout="wide",
)

DATA_DIR = Path("data")

PREDICTIONS_FILE = DATA_DIR / "current_predictions.csv"
METADATA_FILE = DATA_DIR / "model_metadata.json"
HISTORY_FILE = DATA_DIR / "prediction_history.csv"


# ============================================================
# TEAM INFORMATION
# ============================================================

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
    "JAC": "Jaguars",
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
    "WSH": "Commanders",
}


def logo_url(team):
    code = str(team).upper()
    code = ESPN_CODES.get(code, code.lower())

    return (
        f"https://a.espncdn.com/i/"
        f"teamlogos/nfl/500/{code}.png"
    )


# ============================================================
# FORMATTING HELPERS
# ============================================================

def safe_text(value, default="—"):
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    string = str(value).strip()

    if not string or string.lower() in {"nan", "none"}:
        return default

    return string


def fmt_number(value, suffix="", decimals=2):
    try:
        number = float(value)

        if np.isfinite(number):
            return f"{number:.{decimals}f}{suffix}"

    except (ValueError, TypeError):
        pass

    return "—"


def fmt_price(value):
    try:
        number = float(value)

        if np.isfinite(number):
            price = int(number)

            if price > 0:
                return f"+{price}"

            return str(price)

    except (ValueError, TypeError):
        pass

    return "—"


def kickoff_label(row):
    date = pd.to_datetime(
        row.get("gameday"),
        errors="coerce",
    )

    if pd.notna(date):
        date_text = date.strftime("%a, %b %d")
    else:
        date_text = "Date unavailable"

    time_text = safe_text(
        row.get("gametime")
    )

    try:
        clock = pd.to_datetime(
            time_text,
            format="%H:%M",
            errors="raise",
        )

        time_text = (
            clock.strftime("%I:%M %p")
            .lstrip("0")
        )

    except (ValueError, TypeError):
        pass

    return f"{date_text} · {time_text} ET"


def updated_label(frame, metadata):
    raw = metadata.get("generated_at")

    if not raw and "updated_at" in frame.columns:
        raw = frame["updated_at"].dropna().max()

    stamp = pd.to_datetime(
        raw,
        utc=True,
        errors="coerce",
    )

    if pd.isna(stamp):
        return "Time unavailable"

    return (
        stamp
        .tz_convert("America/New_York")
        .strftime("%b %d, %Y · %I:%M %p ET")
    )


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(ttl=120)
def load_predictions():
    if PREDICTIONS_FILE.exists():
        return pd.read_csv(PREDICTIONS_FILE)

    return None


@st.cache_data(ttl=120)
def load_metadata():
    if METADATA_FILE.exists():

        with METADATA_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    return {}


@st.cache_data(ttl=120)
def load_history():
    if HISTORY_FILE.exists():
        return pd.read_csv(HISTORY_FILE)

    return None


# ============================================================
# CUSTOM WEBSITE DESIGN
# ============================================================

st.markdown(
    """
<style>

.block-container {
    max-width: 1280px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}

.dashboard-hero {
    background: linear-gradient(
        120deg,
        #101e35 0%,
        #173956 55%,
        #125e68 100%
    );

    border: 1px solid rgba(147,197,253,.20);
    border-radius: 18px;
    color: #f8fafc;
    padding: 24px 28px 22px;
    margin-bottom: 14px;
}

.dashboard-eyebrow {
    color: #82e8de;
    font-size: .78rem;
    letter-spacing: .15em;
    font-weight: 700;
}

.dashboard-title {
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1.2;
    margin: 6px 0 8px;
}

.dashboard-sub {
    color: #cbd5e1;
    font-size: .96rem;
    margin: 0;
}

.signal-chip {
    display: inline-block;
    border-radius: 20px;
    padding: 4px 10px;
    font-size: .76rem;
    font-weight: 750;
    letter-spacing: .04em;
}

.chip-high {
    background: #dcfce7;
    color: #166534;
}

.chip-strong {
    background: #cffafe;
    color: #155e75;
}

.chip-standard {
    background: #fef3c7;
    color: #92400e;
}

.detail-label {
    color: #8894a5;
    font-size: .77rem;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: .04em;
    margin-bottom: 2px;
}

div[data-testid="stMetric"] {
    border: 1px solid rgba(140,150,170,.20);
    padding: 12px 16px;
    border-radius: 12px;
    background: rgba(120,140,160,.045);
}

[data-testid="stMetricValue"] {
    font-size: 1.55rem;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# WEBSITE HEADER
# ============================================================

st.html("""
<div class="dashboard-hero">
    <div class="dashboard-eyebrow">
        NFL · SPORTS ANALYTICS
    </div>

    <div class="dashboard-title">
        NFL Spread Lab
    </div>

    <p class="dashboard-sub">
        EPA predictions · sportsbook consensus ·
        best available lines
    </p>
</div>
""")


# ============================================================
# READ PREDICTIONS
# ============================================================

predictions = load_predictions()
metadata = load_metadata()

if predictions is None:
    st.error(
        "No prediction file found. "
        "Run `python build_predictions.py` first."
    )
    st.stop()

if predictions.empty:
    st.info(
        "No games in the latest prediction file."
    )
    st.stop()


# ============================================================
# VALIDATE DATA
# ============================================================

required_columns = {
    "season",
    "week",
    "gameday",
    "gametime",
    "away_team",
    "home_team",
    "model_pick",
    "signal",
    "edge",
}

missing = required_columns.difference(
    predictions.columns
)

if missing:
    st.error(
        "Prediction data is missing: "
        + ", ".join(sorted(missing))
    )
    st.stop()


# ============================================================
# CLEAN NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "season",
    "week",
    "edge",
    "abs_edge",
    "best_line_edge",
    "best_price",
    "book_count",
]

for column in numeric_columns:

    if column in predictions.columns:

        predictions[column] = pd.to_numeric(
            predictions[column],
            errors="coerce",
        )


if "abs_edge" not in predictions.columns:

    predictions["abs_edge"] = (
        predictions["edge"].abs()
    )


# ============================================================
# NFL WEEK AND KICKOFF TIMES
# ============================================================

predictions["_kickoff"] = pd.to_datetime(
    predictions["gameday"].astype(str)
    + " "
    + predictions["gametime"].astype(str),
    errors="coerce",
)

season = int(
    predictions["season"].dropna().iloc[0]
)

week = int(
    predictions["week"].dropna().iloc[0]
)

threshold = float(
    metadata.get("edge_threshold", 4.0)
)


# ============================================================
# HIDE GAMES THAT HAVE ALREADY STARTED
# ============================================================

# NFL schedule times are Eastern local times.

now_et = (
    pd.Timestamp.now(
        tz="America/New_York"
    )
    .tz_localize(None)
)

upcoming = predictions.loc[
    predictions["_kickoff"].notna()
    & predictions["_kickoff"].gt(now_et)
].copy()


current_signals = upcoming.loc[
    upcoming["signal"].eq("4+ EDGE")
].copy()


# ============================================================
# LOAD OFFICIAL FROZEN PREDICTIONS
# ============================================================

frozen_week = pd.DataFrame()
frozen_signals = pd.DataFrame()

history_error = None

try:
    history = load_history()

    needed = {
        "season",
        "week",
        "away_team",
        "home_team",
        "model_pick",
        "signal",
        "consensus_edge",
    }

    if (
        history is not None
        and not history.empty
        and needed.issubset(history.columns)
    ):

        history["season"] = pd.to_numeric(
            history["season"],
            errors="coerce",
        )

        history["week"] = pd.to_numeric(
            history["week"],
            errors="coerce",
        )

        frozen_week = history.loc[
            history["season"].eq(season)
            & history["week"].eq(week)
        ].copy()

        frozen_signals = frozen_week.loc[
            frozen_week["signal"].eq("4+ EDGE")
        ].copy()

except (
    OSError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
) as error:

    history_error = str(error)


# ============================================================
# COUNT SPORTSBOOKS
# ============================================================

if (
    "book_count" in upcoming.columns
    and upcoming["book_count"].notna().any()
):

    books = int(
        upcoming["book_count"].max()
    )

else:
    books = 0


# ============================================================
# LAST UPDATE
# ============================================================

st.caption(
    f"Latest saved odds snapshot: "
    f"{updated_label(predictions, metadata)}"
    " · Not continuously streaming prices"
)


# ============================================================
# MAIN DASHBOARD METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "NFL Slate",
    f"{season} · Week {week}",
)

m2.metric(
    "Upcoming Games",
    len(upcoming),
)

m3.metric(
    f"Upcoming {threshold:g}+ Edges",
    len(current_signals),
)

m4.metric(
    "Locked 4+ Signals This Week",
    len(frozen_signals),
)


st.caption(
    f"{books} sportsbooks represented in the "
    "upcoming snapshot. Current edges may move; "
    "locked predictions do not change. "
    "The locked count includes games already "
    "kicked off this week."
)


# ============================================================
# DATA WARNINGS
# ============================================================

if history_error:

    st.warning(
        f"Could not load prediction history: "
        f"{history_error}"
    )

elif not HISTORY_FILE.exists():

    st.info(
        "No locked prediction history file yet. "
        "The tracker will create it after the "
        "next successful update."
    )


if "line_source" in upcoming.columns:

    fallbacks = int(
        upcoming["line_source"]
        .ne("Live consensus")
        .sum()
    )

    if fallbacks:

        st.warning(
            f"{fallbacks} matchup(s) use stored "
            "nflverse fallback lines."
        )


# ============================================================
# CURRENT VS LOCKED COMPARISON
# ============================================================

with st.expander(
    "Compare Upcoming vs. Locked Predictions",
    expanded=False,
):

    st.caption(
        "Current = latest saved sportsbook snapshot "
        "for games not yet started. "
        "Locked = first pre-kickoff prediction "
        "saved by the results tracker."
    )

    if upcoming.empty:

        st.info(
            "No upcoming games left to compare "
            "this week."
        )

    else:

        comparison = upcoming[
            [
                "away_team",
                "home_team",
                "model_pick",
                "signal",
                "abs_edge",
            ]
        ].copy()

        comparison.columns = [
            "Away",
            "Home",
            "Current Pick",
            "Current Signal",
            "Current Edge",
        ]

        # --------------------------------
        # ADD LOCKED PREDICTIONS
        # --------------------------------

        if not frozen_week.empty:

            locked = frozen_week[
                [
                    "away_team",
                    "home_team",
                    "model_pick",
                    "signal",
                    "consensus_edge",
                ]
            ].copy()

            locked = locked.drop_duplicates(
                subset=[
                    "away_team",
                    "home_team",
                ],
                keep="first",
            )

            locked.columns = [
                "Away",
                "Home",
                "Locked Pick",
                "Locked Signal",
                "Locked Edge",
            ]

            comparison = comparison.merge(
                locked,
                on=[
                    "Away",
                    "Home",
                ],
                how="left",
            )

        else:

            comparison["Locked Pick"] = pd.NA
            comparison["Locked Signal"] = pd.NA
            comparison["Locked Edge"] = np.nan

        # --------------------------------
        # CLEAN COMPARISON
        # --------------------------------

        for column in [
            "Current Edge",
            "Locked Edge",
        ]:

            comparison[column] = (
                pd.to_numeric(
                    comparison[column],
                    errors="coerce",
                )
                .abs()
                .round(2)
            )

        comparison["Locked Pick"] = (
            comparison["Locked Pick"]
            .fillna("Not locked")
        )

        comparison["Locked Signal"] = (
            comparison["Locked Signal"]
            .fillna("Not locked")
        )

        comparison = comparison[
            [
                "Away",
                "Home",
                "Current Pick",
                "Locked Pick",
                "Current Signal",
                "Locked Signal",
                "Current Edge",
                "Locked Edge",
            ]
        ]

        st.dataframe(
            comparison,
            hide_index=True,
            use_container_width=True,
        )


# ============================================================
# TOP MODEL EDGES
# ============================================================

st.divider()

st.subheader(
    "🔥 Top Model Edges"
)

st.caption(
    "Upcoming 4+ consensus signals only. "
    "Lines and prices come from the latest "
    "saved sportsbook snapshot."
)


if current_signals.empty:

    st.info(
        "No upcoming 4+ edge signals right now."
    )

else:

    current_signals = (
        current_signals.sort_values(
            "abs_edge",
            ascending=False,
        )
    )

    for _, game in current_signals.iterrows():

        away = safe_text(
            game.get("away_team")
        )

        home = safe_text(
            game.get("home_team")
        )

        if pd.notna(
            game.get("abs_edge")
        ):
            edge = float(
                game["abs_edge"]
            )
        else:
            edge = 0.0

        # --------------------------------
        # SIGNAL STRENGTH
        # --------------------------------

        if edge >= 7:

            chip_class = "chip-high"
            chip_name = "HIGH EDGE"

        elif edge >= 5:

            chip_class = "chip-strong"
            chip_name = "STRONG EDGE"

        else:

            chip_class = "chip-standard"
            chip_name = "4+ EDGE"

        # --------------------------------
        # MATCHUP CARD
        # --------------------------------

        with st.container(border=True):

            (
                a_logo,
                a_team,
                h_logo,
                h_team,
                edge_metric,
            ) = st.columns(
                [0.8, 2.0, 0.8, 2.0, 1.65],
                vertical_alignment="center",
                gap="small",
            )

            # AWAY TEAM

            with a_logo:

                st.image(
                    logo_url(away),
                    width=60,
                )

            with a_team:

                st.markdown(
                    f"**{away} · "
                    f"{TEAM_NAMES.get(away, away)}**"
                )

                st.caption("AWAY")

            # HOME TEAM

            with h_logo:

                st.image(
                    logo_url(home),
                    width=60,
                )

            with h_team:

                st.markdown(
                    f"**{home} · "
                    f"{TEAM_NAMES.get(home, home)}**"
                )

                st.caption(
                    f"HOME · {kickoff_label(game)}"
                )

            # EDGE

            with edge_metric:

                st.metric(
                    "Consensus Edge",
                    f"{edge:.2f} pts",
                )

                st.markdown(
                    f'<span class="signal-chip '
                    f'{chip_class}">'
                    f'{chip_name}</span>',
                    unsafe_allow_html=True,
                )

            st.divider()

            # --------------------------------
            # SPREAD INFORMATION
            # --------------------------------

            c1, c2, c3 = st.columns(
                [1, 1, 1.35],
                gap="medium",
            )

            with c1:

                st.markdown(
                    '<div class="detail-label">'
                    'Consensus Spread</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"**{safe_text(game.get('market_line'))}**"
                )

                st.markdown(
                    '<div class="detail-label">'
                    'Model Fair Spread</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"**{safe_text(game.get('model_fair_spread'))}**"
                )

            with c2:

                st.markdown(
                    '<div class="detail-label">'
                    'Model Selection</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"**{safe_text(game.get('model_pick'))}**"
                )

                st.markdown(
                    '<div class="detail-label">'
                    'Best-Line Edge</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"**{fmt_number(game.get('best_line_edge'), ' pts')}**"
                )

            with c3:

                st.markdown(
                    '<div class="detail-label">'
                    'Best Available Sportsbook Line</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"**{safe_text(game.get('best_available_line'))}**"
                )

                st.markdown(
                    f"{safe_text(game.get('best_bookmaker'))}"
                    f" · **{fmt_price(game.get('best_price'))}**"
                    " American odds"
                )

                st.caption(
                    f"Source: "
                    f"{safe_text(game.get('line_source'))}"
                    " · Saved snapshot"
                )


# ============================================================
# FULL WEEKLY BOARD
# ============================================================

st.divider()

st.subheader(
    "📊 Full Weekly Board"
)


if upcoming.empty:

    st.info(
        "There are no remaining upcoming games "
        "in this saved slate."
    )

else:

    # --------------------------------
    # TEAM FILTER
    # --------------------------------

    team_list = sorted(
        set(upcoming["away_team"].dropna())
        |
        set(upcoming["home_team"].dropna())
    )

    f1, f2, f3 = st.columns(
        [1.4, 1.1, 1.5]
    )

    with f1:

        team_filter = st.selectbox(
            "Team",
            ["All teams"] + team_list,
        )

    with f2:

        signal_only = st.checkbox(
            "4+ Signals Only",
            value=False,
        )

    with f3:

        sort_choice = st.selectbox(
            "Sort",
            [
                "Largest Edge",
                "Game Time",
                "Best-Line Edge",
            ],
        )

    # --------------------------------
    # APPLY FILTERS
    # --------------------------------

    board = upcoming.copy()

    if team_filter != "All teams":

        board = board.loc[
            board["home_team"].eq(team_filter)
            |
            board["away_team"].eq(team_filter)
        ]

    if signal_only:

        board = board.loc[
            board["signal"].eq("4+ EDGE")
        ]

    # --------------------------------
    # SORT
    # --------------------------------

    if sort_choice == "Game Time":

        board = board.sort_values(
            "_kickoff",
            na_position="last",
        )

    elif (
        sort_choice == "Best-Line Edge"
        and "best_line_edge" in board.columns
    ):

        board = board.sort_values(
            "best_line_edge",
            ascending=False,
            na_position="last",
        )

    else:

        board = board.sort_values(
            "abs_edge",
            ascending=False,
            na_position="last",
        )

    # --------------------------------
    # DISPLAY
    # --------------------------------

    if board.empty:

        st.info(
            "No games match your filters."
        )

    else:

        display_columns = [
            "gameday",
            "gametime",
            "away_team",
            "home_team",
            "market_line",
            "model_fair_spread",
            "abs_edge",
            "model_pick",
            "signal",
            "best_available_line",
            "best_bookmaker",
            "best_price",
            "best_line_edge",
            "line_source",
            "book_count",
        ]

        board = board[
            [
                c
                for c in display_columns
                if c in board.columns
            ]
        ].copy()

        # ROUND EDGES

        for column in [
            "abs_edge",
            "best_line_edge",
        ]:

            if column in board.columns:

                board[column] = (
                    pd.to_numeric(
                        board[column],
                        errors="coerce",
                    )
                    .round(2)
                )

        # FORMAT PRICES

        if "best_price" in board.columns:

            board["best_price"] = (
                board["best_price"]
                .map(fmt_price)
            )

        # FORMAT SIGNAL

        if "signal" in board.columns:

            board["signal"] = (
                board["signal"]
                .replace({
                    "4+ EDGE": "🔥 4+ EDGE"
                })
            )

        # RENAME COLUMNS

        board.rename(
            columns={
                "gameday": "Date",
                "gametime": "ET",
                "away_team": "Away",
                "home_team": "Home",
                "market_line": "Consensus",
                "model_fair_spread": "Model Fair",
                "abs_edge": "Edge (pts)",
                "model_pick": "Model Pick",
                "signal": "Signal",
                "best_available_line": "Best Line",
                "best_bookmaker": "Sportsbook",
                "best_price": "Price",
                "best_line_edge": "Best Edge (pts)",
                "line_source": "Line Source",
                "book_count": "Books",
            },
            inplace=True,
        )

        st.dataframe(
            board,
            use_container_width=True,
            hide_index=True,
            height=min(
                620,
                45 + 37 * len(board),
            ),
        )

        st.caption(
            f"Showing {len(board)} of "
            f"{len(upcoming)} upcoming games. "
            "Scroll horizontally for all odds columns."
        )


# ============================================================
# MODEL RESULTS PAGE LINK
# ============================================================

st.write("")

if Path(
    "pages/1_Model_Results.py"
).exists():

    st.page_link(
        "pages/1_Model_Results.py",
        label="View Your Tracked Model Results",
        icon="📊",
    )


# ============================================================
# MODEL DETAILS
# ============================================================

with st.expander(
    "Model Details & Historical Walk-Forward Testing",
    expanded=False,
):

    st.write(
        f"**Model:** "
        f"{safe_text(metadata.get('model', 'StandardScaler + Ridge(alpha=10)'))}"
    )

    st.write(
        f"**Training Games:** "
        f"{safe_text(metadata.get('training_games'))}"
    )

    st.write(
        f"**Official Signal Rule:** "
        f"{threshold:g}+ points "
        "from the saved consensus line"
    )

    backtest = metadata.get(
        "historical_walk_forward_2021_2025",
        {},
    )

    if backtest:

        x1, x2, x3, x4 = st.columns(4)

        x1.metric(
            "Historical Bets",
            safe_text(backtest.get("bets")),
        )

        x2.metric(
            "Historical Record",
            safe_text(backtest.get("record")),
        )

        x3.metric(
            "Historical Win Rate",
            fmt_number(
                backtest.get("win_rate_pct"),
                "%",
            ),
        )

        x4.metric(
            "Historical ROI",
            fmt_number(
                backtest.get("roi_pct"),
                "%",
            ),
        )

        st.caption(
            "Historical development backtest only. "
            "These results are separate from "
            "the forward-tracked predictions."
        )

    st.markdown(
        "**How to read the numbers:** "
        "Consensus is the median available sportsbook "
        "spread when the snapshot was generated. "
        "The model fair line is a regression prediction. "
        "An edge represents a difference in points, "
        "not a calibrated probability. "
        "The best line and price are recorded offers, "
        "not continuously refreshed quotes."
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "Educational sports analytics project. "
    "Sportsbook odds are recorded snapshots and "
    "may change. Historical performance does "
    "not guarantee future results."
)
