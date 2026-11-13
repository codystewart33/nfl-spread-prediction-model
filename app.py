"""NFL spread model dashboard: polished presentation of generated model data.

Reads the existing CSV/metadata. Does not retrain, fetch odds, or change signals.
The existing pages/1_Model_Results.py continues to work as a separate page.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st


# --------------------------- PAGE & ASSETS ---------------------------
st.set_page_config(page_title="NFL Spread Lab", page_icon="🏈", layout="wide")

DATA_DIR = Path("data")
PREDICTIONS_FILE = DATA_DIR / "current_predictions.csv"
METADATA_FILE = DATA_DIR / "model_metadata.json"

# ESPN uses a few different abbreviations than nflverse.
ESPN_CODES = {"LA": "lar", "WAS": "wsh", "JAC": "jax", "WSH": "wsh"}

TEAM_NAMES = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills",
    "CAR": "Panthers", "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns",
    "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
    "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "JAC": "Jaguars",
    "KC": "Chiefs", "LA": "Rams", "LAC": "Chargers", "LV": "Raiders",
    "MIA": "Dolphins", "MIN": "Vikings", "NE": "Patriots", "NO": "Saints",
    "NYG": "Giants", "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers",
    "SEA": "Seahawks", "SF": "49ers", "TB": "Buccaneers", "TEN": "Titans",
    "WAS": "Commanders", "WSH": "Commanders",
}


def logo_url(team):
    code = str(team).upper()
    code = ESPN_CODES.get(code, code.lower())
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{code}.png"


def safe_text(value, default="—"):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    return s if s and s.lower() not in ("nan", "none") else default


def fmt_number(value, suffix="", decimals=2):
    try:
        n = float(value)
        if not np.isfinite(n):
            return "—"
        return f"{n:.{decimals}f}{suffix}"
    except (ValueError, TypeError):
        return "—"


def fmt_price(value):
    try:
        n = float(value)
        if not np.isfinite(n):
            return "—"
        price = int(n)
        return f"+{price}" if price > 0 else str(price)
    except (ValueError, TypeError):
        return "—"


def kickoff_label(row):
    raw_date = row.get("gameday")
    raw_time = row.get("gametime")
    parsed = pd.to_datetime(raw_date, errors="coerce")
    date = f"{parsed.strftime('%a, %b')} {parsed.day}" if pd.notna(parsed) else safe_text(raw_date)
    time = safe_text(raw_time)
    try:
        t = pd.to_datetime(time, format="%H:%M", errors="raise")
        time = t.strftime("%I:%M %p").lstrip("0")
    except (TypeError, ValueError):
        pass
    return f"{date} · {time} ET"


def updated_label(df, meta):
    raw = meta.get("generated_at")
    if not raw and "updated_at" in df.columns:
        raw = df["updated_at"].dropna().max()
    if raw is None:
        return "Snapshot time unavailable"
    stamp = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(stamp):
        return "Snapshot time unavailable"
    return stamp.tz_convert("America/New_York").strftime("%b %d, %Y · %I:%M %p ET")


@st.cache_data(ttl=120)
def load_predictions():
    if not PREDICTIONS_FILE.exists():
        return None
    return pd.read_csv(PREDICTIONS_FILE)


@st.cache_data(ttl=120)
def load_metadata():
    if not METADATA_FILE.exists():
        return {}
    with METADATA_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------- VISUAL THEME ---------------------------
st.markdown(
    """<style>
    .block-container {max-width: 1280px; padding-top: 1.5rem; padding-bottom: 4rem;}
    .dashboard-hero {
        background: linear-gradient(120deg, #101e35 0%, #173956 55%, #125e68 100%);
        border: 1px solid rgba(147,197,253,.20); border-radius: 18px;
        color: #f8fafc; padding: 24px 28px 22px; margin-bottom: 14px;
        box-shadow: 0 12px 35px rgba(0,0,0,.10);
    }
    .dashboard-eyebrow {color: #82e8de; font-size: .78rem; letter-spacing: .15em; font-weight: 700;}
    .dashboard-title {font-size: 2.1rem; font-weight: 800; line-height: 1.2; margin: 6px 0 8px;}
    .dashboard-sub {color: #cbd5e1; font-size: .96rem; margin: 0;}
    .signal-chip {display:inline-block; border-radius: 20px; padding: 4px 10px;
        font-size: .76rem; font-weight: 750; letter-spacing: .04em;}
    .chip-high {background: #dcfce7; color: #166534;}
    .chip-strong {background: #cffafe; color: #155e75;}
    .chip-standard {background: #fef3c7; color: #92400e;}
    .detail-label {color: #8894a5; font-size: .77rem; font-weight: 650;
        text-transform: uppercase; letter-spacing: .04em; margin-bottom: 2px;}
    div[data-testid="stMetric"] {border: 1px solid rgba(140,150,170,.20);
        padding: 12px 16px; border-radius: 12px; background: rgba(120,140,160,.045);}
    [data-testid="stMetricValue"] {font-size: 1.55rem;}
    </style>""",
    unsafe_allow_html=True,
)

predictions = load_predictions()
metadata = load_metadata()

st.markdown(
    """<div class="dashboard-hero">
      <div class="dashboard-eyebrow">NFL · SPORTS ANALYTICS</div>
      <div class="dashboard-title">NFL Spread Lab</div>
      <p class="dashboard-sub">EPA-powered predictions · multi-book consensus · best available sportsbook spreads</p>
    </div>""",
    unsafe_allow_html=True,
)

if predictions is None:
    st.error("No prediction file found. Run `python build_predictions.py` to generate `data/current_predictions.csv`.")
    st.stop()
if predictions.empty:
    st.info("No games are currently in the prediction file. Try your next model update.")
    st.stop()

for col in ("season", "week", "edge", "abs_edge", "best_line_edge", "best_price", "book_count"):
    if col in predictions.columns:
        predictions[col] = pd.to_numeric(predictions[col], errors="coerce")

if "abs_edge" not in predictions.columns and "edge" in predictions.columns:
    predictions["abs_edge"] = predictions["edge"].abs()

if "signal" not in predictions.columns:
    predictions["signal"] = "PASS"

for col in ("gameday", "gametime", "away_team", "home_team"):
    if col not in predictions.columns:
        st.error(f"The predictions file is missing `{col}`. Rebuild the model before using this dashboard.")
        st.stop()

predictions["_kickoff"] = pd.to_datetime(
    predictions["gameday"].astype(str) + " " + predictions["gametime"].astype(str),
    errors="coerce",
)
# Hide games that have already kicked off.
# NFL schedule times are in Eastern Time.

now_et = (
    pd.Timestamp.now(tz="America/New_York")
    .tz_localize(None)
)

predictions = predictions[
    predictions["_kickoff"].notna()
    & (predictions["_kickoff"] > now_et)
].copy()

if predictions.empty:
    st.info("No upcoming games are currently available.")
    st.stop()

season = safe_text(metadata.get("season", predictions["season"].iloc[0] if "season" in predictions else None))
week = safe_text(metadata.get("week", predictions["week"].iloc[0] if "week" in predictions else None))
threshold = float(metadata.get("edge_threshold", 4.0))
signal_rows = predictions[predictions["signal"].eq("4+ EDGE")].copy()
books = int(predictions["book_count"].max()) if "book_count" in predictions and predictions["book_count"].notna().any() else 0
live_rows = int(predictions["line_source"].eq("Live consensus").sum()) if "line_source" in predictions else 0

st.caption(f"Last data snapshot: {updated_label(predictions, metadata)} · Prices may have moved since this update")

m1, m2, m3, m4 = st.columns(4)
m1.metric("NFL slate", f"{season} · Week {week}")
m2.metric("Games on board", len(predictions))
m3.metric(f"{threshold:g}+ point signals", len(signal_rows))
m4.metric("Sportsbooks checked", books)

if live_rows < len(predictions):
    st.warning(
        f"{len(predictions) - live_rows} matchup(s) use a stored nflverse fallback line rather than an odds-feed consensus."
    )

st.write("")
st.subheader("Top model edges")
st.caption("Official signals use the consensus edge. The best sportsbook spread and price are displayed separately.")

if signal_rows.empty:
    st.info("There are no 4+ point signals in the current saved board.")
else:
    for _, game in signal_rows.sort_values("abs_edge", ascending=False).iterrows():
        away = safe_text(game.get("away_team"))
        home = safe_text(game.get("home_team"))
        edge = float(game.get("abs_edge")) if pd.notna(game.get("abs_edge")) else 0.0
        if edge >= 7:
            chip_class, chip_name = "chip-high", "HIGH EDGE"
        elif edge >= 5:
            chip_class, chip_name = "chip-strong", "STRONG EDGE"
        else:
            chip_class, chip_name = "chip-standard", "4+ EDGE"

        with st.container(border=True):
            a_logo, a_team, h_logo, h_team, edge_metric = st.columns(
                [0.8, 2.0, 0.8, 2.0, 1.65], vertical_alignment="center", gap="small"
            )
            with a_logo:
                st.image(logo_url(away), width=60)
            with a_team:
                st.markdown(f"**{away} · {TEAM_NAMES.get(away, away)}**")
                st.caption("AWAY")
            with h_logo:
                st.image(logo_url(home), width=60)
            with h_team:
                st.markdown(f"**{home} · {TEAM_NAMES.get(home, home)}**")
                st.caption(f"HOME · {kickoff_label(game)}")
            with edge_metric:
                st.metric("Consensus edge", f"{edge:.2f} pts")
                st.markdown(
                    f'<span class="signal-chip {chip_class}">{chip_name}</span>',
                    unsafe_allow_html=True,
                )

            st.divider()
            c1, c2, c3 = st.columns([1, 1, 1.35], gap="medium")
            with c1:
                st.markdown('<div class="detail-label">Consensus spread</div>', unsafe_allow_html=True)
                st.markdown(f"**{safe_text(game.get('market_line'))}**")
                st.markdown('<div class="detail-label">Model fair spread</div>', unsafe_allow_html=True)
                st.markdown(f"**{safe_text(game.get('model_fair_spread'))}**")
            with c2:
                st.markdown('<div class="detail-label">Model selection</div>', unsafe_allow_html=True)
                st.markdown(f"**{safe_text(game.get('model_pick'))}**")
                st.markdown('<div class="detail-label">Best-line edge</div>', unsafe_allow_html=True)
                st.markdown(f"**{fmt_number(game.get('best_line_edge'), ' pts')}**")
            with c3:
                st.markdown('<div class="detail-label">Best available sportsbook line</div>', unsafe_allow_html=True)
                st.markdown(f"**{safe_text(game.get('best_available_line'))}**")
                st.markdown(
                    f"{safe_text(game.get('best_bookmaker'))} · "
                    f"**{fmt_price(game.get('best_price'))}** American odds"
                )
                st.caption(f"Source: {safe_text(game.get('line_source'))} · Snapshot, not a live quote")

st.write("")
st.subheader("Full weekly board")

team_list = sorted(set(predictions["away_team"].dropna()) | set(predictions["home_team"].dropna()))
f1, f2, f3 = st.columns([1.4, 1.1, 1.5])
with f1:
    team_filter = st.selectbox("Team", ["All teams"] + team_list)
with f2:
    signal_only = st.checkbox("4+ signals only", value=False)
with f3:
    sort_choice = st.selectbox("Sort", ["Largest edge", "Game time", "Best-line edge"])

board = predictions.copy()
if team_filter != "All teams":
    board = board[(board["home_team"] == team_filter) | (board["away_team"] == team_filter)]
if signal_only:
    board = board[board["signal"] == "4+ EDGE"]

if sort_choice == "Game time":
    board = board.sort_values("_kickoff", na_position="last")
elif sort_choice == "Best-line edge" and "best_line_edge" in board.columns:
    board = board.sort_values("best_line_edge", ascending=False, na_position="last")
else:
    board = board.sort_values("abs_edge", ascending=False, na_position="last")

if board.empty:
    st.info("No games match the current filters.")
else:
    display_cols = [
        "gameday", "gametime", "away_team", "home_team", "market_line",
        "model_fair_spread", "abs_edge", "model_pick", "signal",
        "best_available_line", "best_bookmaker", "best_price",
        "best_line_edge", "line_source", "book_count",
    ]
    board = board[[c for c in display_cols if c in board.columns]].copy()
    for col in ("abs_edge", "best_line_edge"):
        if col in board:
            board[col] = pd.to_numeric(board[col], errors="coerce").round(2)
    if "best_price" in board:
        board["best_price"] = board["best_price"].map(fmt_price)
    if "signal" in board:
        board["signal"] = board["signal"].replace({"4+ EDGE": "🔥 4+ EDGE"})
    board.rename(columns={
        "gameday": "Date", "gametime": "ET", "away_team": "Away", "home_team": "Home",
        "market_line": "Consensus", "model_fair_spread": "Model fair",
        "abs_edge": "Edge (pts)", "model_pick": "Model pick", "signal": "Signal",
        "best_available_line": "Best line", "best_bookmaker": "Sportsbook",
        "best_price": "Price", "best_line_edge": "Best edge (pts)",
        "line_source": "Line source", "book_count": "Books",
    }, inplace=True)
    st.dataframe(board, use_container_width=True, hide_index=True, height=min(620, 45 + 37 * len(board)))
    st.caption(f"Showing {len(board)} of {len(predictions)} games · Scroll horizontally to see all odds columns")

st.write("")
if Path("pages/1_Model_Results.py").exists():
    st.page_link("pages/1_Model_Results.py", label="📈 View your tracked model results", icon="📊")

with st.expander("Model details & historical walk-forward testing", expanded=False):
    st.write(f"**Model:** {safe_text(metadata.get('model', 'StandardScaler + Ridge(alpha=10)'))}")
    st.write(f"**Training games:** {safe_text(metadata.get('training_games'))}")
    st.write(f"**Official signal rule:** {threshold:g}+ points from the saved consensus line")
    history = metadata.get("historical_walk_forward_2021_2025", {})
    if history:
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Historical bets", safe_text(history.get("bets")))
        x2.metric("Historical record", safe_text(history.get("record")))
        x3.metric("Historical win rate", fmt_number(history.get("win_rate_pct"), "%"))
        x4.metric("Historical ROI", fmt_number(history.get("roi_pct"), "%"))
        st.caption("Historical development backtest only; not the same as tracked forward results.")
    st.markdown(
        "**How to read the numbers:** Consensus is the median available spread across books "
        "when the odds snapshot was generated. The model fair line is a regression prediction. "
        "An edge measures a difference in points, **not** a calibrated probability or guarantee. "
        "The best line and price are the most favorable recorded offer for the model's side."
    )

st.divider()
st.caption(
    "Educational sports analytics project. Odds are recorded snapshots, not streaming prices; "
    "books can change or remove lines. Historical performance does not guarantee future results."
)
