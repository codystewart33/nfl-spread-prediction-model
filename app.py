from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="NFL Spread Model",
    page_icon="🏈",
    layout="wide",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #888;
    margin-bottom: 1.5rem;
}

.edge-card {
    border: 1px solid #333;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 14px;
}

.edge-title {
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 8px;
}

.edge-pick {
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 8px;
}

.small-label {
    color: #888;
    font-size: 0.85rem;
}

.best-line {
    font-size: 1rem;
    font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# FILE PATHS
# ============================================================

DATA_DIR = Path("data")

PREDICTIONS_FILE = (
    DATA_DIR / "current_predictions.csv"
)

METADATA_FILE = (
    DATA_DIR / "model_metadata.json"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(ttl=300)
def load_predictions():

    if not PREDICTIONS_FILE.exists():
        return None

    return pd.read_csv(
        PREDICTIONS_FILE
    )


@st.cache_data(ttl=300)
def load_metadata():

    if not METADATA_FILE.exists():
        return {}

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


predictions = load_predictions()
metadata = load_metadata()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🏈 NFL Spread Model'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="subtitle">
EPA-based NFL spread model with live sportsbook
consensus lines and best available prices.
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# NO DATA
# ============================================================

if predictions is None:

    st.error(
        "No prediction file exists yet. "
        "Run python build_predictions.py first."
    )

    st.stop()


if predictions.empty:

    st.warning(
        "The prediction file exists, "
        "but it currently contains no games."
    )

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

numeric_columns = [
    "spread_line",
    "model_margin",
    "edge",
    "abs_edge",
    "best_price",
    "best_line_edge",
    "book_count",
]

for column in numeric_columns:

    if column in predictions.columns:

        predictions[column] = (
            pd.to_numeric(
                predictions[column],
                errors="coerce",
            )
        )


# ============================================================
# TOP METRICS
# ============================================================

season = metadata.get(
    "season",
    predictions["season"].iloc[0]
    if "season" in predictions.columns
    else "—",
)

week = metadata.get(
    "week",
    predictions["week"].iloc[0]
    if "week" in predictions.columns
    else "—",
)

edge_threshold = metadata.get(
    "edge_threshold",
    4.0,
)


if "signal" in predictions.columns:

    signal_count = int(
        (
            predictions["signal"]
            == "4+ EDGE"
        ).sum()
    )

else:

    signal_count = 0


if "book_count" in predictions.columns:

    books_used = predictions[
        "book_count"
    ].max()

    if pd.isna(books_used):
        books_used = 0
    else:
        books_used = int(books_used)

else:

    books_used = 0


col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "NFL Week",
        f"{season} Week {week}",
    )

with col2:

    st.metric(
        "Upcoming Games",
        len(predictions),
    )

with col3:

    st.metric(
        "4+ Edge Signals",
        signal_count,
    )

with col4:

    st.metric(
        "Sportsbooks",
        books_used,
    )


st.divider()


# ============================================================
# CURRENT MODEL SIGNALS
# ============================================================

st.subheader(
    "🔥 Current Model Signals"
)

st.caption(
    "Signals are based on the model's difference "
    "from the live consensus spread. "
    "The best available sportsbook line is shown separately."
)


if "signal" in predictions.columns:

    edges = predictions[
        predictions["signal"]
        == "4+ EDGE"
    ].copy()

else:

    edges = pd.DataFrame()


if edges.empty:

    st.info(
        "There are currently no 4+ point "
        "model edges."
    )

else:

    sort_column = (
        "abs_edge"
        if "abs_edge" in edges.columns
        else "edge"
    )

    edges = edges.sort_values(
        sort_column,
        ascending=False,
    )

    for _, row in edges.iterrows():

        away = row.get(
            "away_team",
            "",
        )

        home = row.get(
            "home_team",
            "",
        )

        market = row.get(
            "market_line",
            "—",
        )

        fair = row.get(
            "model_fair_spread",
            "—",
        )

        pick = row.get(
            "model_pick",
            "—",
        )

        edge = row.get(
            "abs_edge",
            np.nan,
        )

        best_line = row.get(
            "best_available_line",
            "—",
        )

        best_book = row.get(
            "best_bookmaker",
            "—",
        )

        best_price = row.get(
            "best_price",
            np.nan,
        )

        best_line_edge = row.get(
            "best_line_edge",
            np.nan,
        )

        game_date = row.get(
            "gameday",
            "",
        )

        game_time = row.get(
            "gametime",
            "",
        )

        # -----------------------
        # FORMAT EDGE
        # -----------------------

        if pd.notna(edge):

            edge_text = (
                f"{edge:.2f} pts"
            )

        else:

            edge_text = "—"

        # -----------------------
        # FORMAT PRICE
        # -----------------------

        if pd.notna(best_price):

            price_int = int(
                best_price
            )

            if price_int > 0:

                price_text = (
                    f"+{price_int}"
                )

            else:

                price_text = (
                    str(price_int)
                )

        else:

            price_text = "—"

        # -----------------------
        # FORMAT BEST EDGE
        # -----------------------

        if pd.notna(
            best_line_edge
        ):

            best_edge_text = (
                f"{best_line_edge:.2f} pts"
            )

        else:

            best_edge_text = "—"

        # -----------------------
        # FIXED HTML CARD
        # -----------------------

        card_html = f"""<div class="edge-card">
<div class="edge-title">{away} @ {home}</div>
<div class="small-label">{game_date} {game_time}</div>
<br>
<div class="edge-pick">Model Pick: {pick}</div>
<div>Consensus Market: <strong>{market}</strong></div>
<div>Model Fair Line: <strong>{fair}</strong></div>
<div>Consensus Edge: <strong>{edge_text}</strong></div>
<br>
<div class="small-label">BEST AVAILABLE</div>
<div class="best-line">{best_line} at {best_book} ({price_text})</div>
<div>Edge at Best Line: <strong>{best_edge_text}</strong></div>
</div>"""

        st.markdown(
            card_html,
            unsafe_allow_html=True,
        )


st.divider()


# ============================================================
# FULL WEEKLY BOARD
# ============================================================

st.subheader(
    "📊 Full Weekly Board"
)

st.caption(
    "Consensus edge determines the official model signal. "
    "Best available line shows where the strongest current "
    "number is available for the model side."
)


display = predictions.copy()


# ============================================================
# TABLE COLUMN NAMES
# ============================================================

rename_map = {

    "gameday":
        "Date",

    "gametime":
        "Time",

    "away_team":
        "Away",

    "home_team":
        "Home",

    "market_line":
        "Consensus",

    "model_fair_spread":
        "Model Fair",

    "edge":
        "Raw Edge",

    "abs_edge":
        "Edge",

    "model_pick":
        "Model Pick",

    "signal":
        "Signal",

    "best_available_line":
        "Best Line",

    "best_bookmaker":
        "Sportsbook",

    "best_price":
        "Price",

    "best_line_edge":
        "Best-Line Edge",

    "book_count":
        "Books",

    "line_source":
        "Line Source",
}


table_columns = [
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
    "book_count",
]


table_columns = [
    c
    for c in table_columns
    if c in display.columns
]


display = display[
    table_columns
].rename(
    columns=rename_map
)


# ============================================================
# ROUND NUMBERS
# ============================================================

if "Edge" in display.columns:

    display["Edge"] = (
        display["Edge"]
        .round(2)
    )


if "Best-Line Edge" in display.columns:

    display[
        "Best-Line Edge"
    ] = (
        display[
            "Best-Line Edge"
        ]
        .round(2)
    )


# ============================================================
# FORMAT AMERICAN ODDS
# ============================================================

if "Price" in display.columns:

    def format_price(value):

        if pd.isna(value):
            return "—"

        value = int(value)

        if value > 0:

            return (
                f"+{value}"
            )

        return str(value)

    display[
        "Price"
    ] = (
        display[
            "Price"
        ].apply(
            format_price
        )
    )


# ============================================================
# SHOW TABLE
# ============================================================

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
)


st.divider()


# ============================================================
# HOW TO READ THE MODEL
# ============================================================

st.subheader(
    "🧠 How to Read the Model"
)

st.markdown(
    """
**Consensus**

Median spread across the sportsbooks currently returned
by the odds feed.

**Model Fair**

The spread predicted by the EPA-based Ridge regression model.

**Edge**

Difference between the model fair spread and the consensus
market spread.

**4+ EDGE**

The model differs from the consensus market by at least
4 points.

**Best Line**

The most favorable spread currently available for the
model-selected side.

**Sportsbook**

The sportsbook currently offering that best spread.

**Price**

American odds attached to that spread.

**Best-Line Edge**

The model's spread difference using the best available
sportsbook number instead of the consensus line.
"""
)


st.divider()


# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader(
    "⚙️ Model"
)


model_name = metadata.get(
    "model",
    "StandardScaler + Ridge(alpha=10)",
)

training_games = metadata.get(
    "training_games",
    "—",
)

generated_at = metadata.get(
    "generated_at",
    "—",
)


model_col1, model_col2, model_col3 = (
    st.columns(3)
)

with model_col1:

    st.metric(
        "Model",
        model_name,
    )

with model_col2:

    st.metric(
        "Training Games",
        training_games,
    )

with model_col3:

    st.metric(
        "Edge Threshold",
        f"{edge_threshold:g} points",
    )


st.caption(
    f"Last model refresh: "
    f"{generated_at}"
)


# ============================================================
# HISTORICAL WALK-FORWARD RESULTS
# ============================================================

history = metadata.get(
    "historical_walk_forward_2021_2025",
    {},
)


if history:

    st.subheader(
        "📈 Historical Walk-Forward Test"
    )

    h1, h2, h3, h4 = (
        st.columns(4)
    )

    with h1:

        st.metric(
            "Bets",
            history.get(
                "bets",
                "—",
            ),
        )

    with h2:

        st.metric(
            "Record",
            history.get(
                "record",
                "—",
            ),
        )

    with h3:

        win_rate = history.get(
            "win_rate_pct"
        )

        if win_rate is not None:

            st.metric(
                "Win Rate",
                f"{win_rate:.2f}%",
            )

        else:

            st.metric(
                "Win Rate",
                "—",
            )

    with h4:

        roi = history.get(
            "roi_pct"
        )

        if roi is not None:

            st.metric(
                "ROI",
                f"{roi:.2f}%",
            )

        else:

            st.metric(
                "ROI",
                "—",
            )

    profit = history.get(
        "profit_units_at_minus_110"
    )

    if profit is not None:

        st.caption(
            f"Historical profit at -110: "
            f"{profit:+.2f} units"
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "This project is for analytical and educational purposes. "
    "Sportsbook lines can move at any time, and historical "
    "results do not guarantee future performance."
)