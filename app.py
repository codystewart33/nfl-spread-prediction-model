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