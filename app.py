from pathlib import Path
import json

import pandas as pd
import streamlit as st


DATA_PATH = Path("data/current_predictions.csv")
META_PATH = Path("data/model_metadata.json")

st.set_page_config(
    page_title="NFL Spread Model",
    page_icon="🏈",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {max-width: 1250px; padding-top: 2rem;}
        .hero {
            padding: 1.2rem 1.4rem;
            border: 1px solid rgba(128,128,128,.25);
            border-radius: 16px;
            margin-bottom: 1rem;
        }
        .muted {opacity: .72;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1 style="margin-bottom:.25rem;">🏈 NFL Spread Model</h1>
      <div class="muted">
        Pass/rush EPA + success-rate model with a 4-point market-disagreement signal.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not DATA_PATH.exists():
    st.error(
        "No prediction file exists yet. Run `python build_predictions.py` first."
    )
    st.stop()

df = pd.read_csv(DATA_PATH)

metadata = {}
if META_PATH.exists():
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))

season = metadata.get("season", df["season"].iloc[0] if "season" in df else "—")
week = metadata.get("week", df["week"].iloc[0] if "week" in df else "—")
generated = metadata.get("generated_at", "—")

signals = df[df["signal"] == "4+ EDGE"].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("NFL Week", f"{season} • Week {week}")
c2.metric("Upcoming Games", len(df))
c3.metric("4+ Edge Signals", len(signals))
c4.metric("Signal Threshold", "4.0 pts")

st.caption(f"Last model refresh: {generated}")

st.divider()

st.subheader("🔥 Current Model Signals")

if signals.empty:
    st.info("No 4+ point model disagreements are currently on the board.")
else:
    signal_view = signals[
        [
            "away_team",
            "home_team",
            "market_line",
            "model_fair_spread",
            "edge",
            "model_pick",
        ]
    ].copy()

    signal_view.columns = [
        "Away",
        "Home",
        "Market",
        "Model Fair Spread",
        "Edge",
        "Model Side",
    ]

    st.dataframe(
        signal_view.style.format({"Edge": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

st.subheader("📊 Full Weekly Board")

board_cols = [
    "away_team",
    "home_team",
    "market_line",
    "model_fair_spread",
    "model_margin",
    "edge",
    "model_pick",
    "signal",
]

board = df[[c for c in board_cols if c in df.columns]].copy()
board = board.rename(columns={
    "away_team": "Away",
    "home_team": "Home",
    "market_line": "Market",
    "model_fair_spread": "Model Fair Spread",
    "model_margin": "Model Margin",
    "edge": "Edge",
    "model_pick": "Model Side",
    "signal": "Signal",
})

formatters = {}
if "Model Margin" in board.columns:
    formatters["Model Margin"] = "{:.2f}"
if "Edge" in board.columns:
    formatters["Edge"] = "{:.2f}"

st.dataframe(
    board.style.format(formatters),
    use_container_width=True,
    hide_index=True,
)

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("📈 Historical Development Backtest")
    hist = metadata.get("historical_walk_forward_2021_2025", {})
    if hist:
        a, b = st.columns(2)
        a.metric("2021–25 ATS", hist.get("record", "—"))
        b.metric("Win Rate", f"{hist.get('win_rate_pct', 0):.2f}%")
        c, d = st.columns(2)
        c.metric("Profit at -110", f"{hist.get('profit_units_at_minus_110', 0):.2f} u")
        d.metric("ROI", f"{hist.get('roi_pct', 0):.2f}%")

with right:
    st.subheader("🧠 Model")
    st.write(
        """
        The model predicts **home-team scoring margin** from trailing five-game
        passing/rushing EPA and success-rate differences for offense and defense.
        Features are standardized and fit with Ridge regression.
        """
    )
    st.write(
        """
        A **4+ EDGE** means the model's fair spread differs from the stored market
        reference line by at least four points. It is a model signal, not a guarantee.
        """
    )

st.caption(
    "Historical performance does not guarantee future results. "
    "The stored market line may differ from the line currently available at a sportsbook."
)
