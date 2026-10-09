# NFL Spread Model Dashboard

A Streamlit dashboard for an NFL point-spread model built from nflverse/nflreadpy data.

## Model

The production model uses:

- trailing five-game passing EPA/play
- trailing five-game rushing EPA/play
- passing success rate
- rushing success rate
- defensive passing EPA allowed
- defensive rushing EPA allowed
- defensive passing success rate allowed
- defensive rushing success rate allowed
- StandardScaler
- Ridge regression (`alpha=10`)

A `4+ EDGE` is shown when the model fair margin differs from the stored market reference line by at least 4 points.

## Run locally

```bash
python -m pip install -r requirements.txt
python build_predictions.py
streamlit run app.py
```

## Project structure

```text
.
├── app.py
├── build_predictions.py
├── requirements.txt
├── data/
│   ├── current_predictions.csv
│   └── model_metadata.json
└── .github/
    └── workflows/
        └── update_model.yml
```

## Automatic refresh

The included GitHub Actions workflow runs every Tuesday morning in the
`America/New_York` timezone and rebuilds the predictions after the prior NFL
week has finished.

You can also run it manually from the GitHub Actions tab using
`workflow_dispatch`.

## Deploy with Streamlit Community Cloud

1. Put this project in a GitHub repository.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Create a new app.
4. Select the repository and `app.py`.
5. Deploy.

When GitHub Actions commits a refreshed prediction CSV, Streamlit will read the
updated file from the repository.

## Important notes

- `spread_line` comes from the schedule data and should be treated as a market
  reference line, not necessarily a real-time sportsbook quote.
- The app intentionally excludes games in the target week from the model's
  training/features, so a Thursday game cannot leak into Sunday predictions.
- Historical results do not guarantee future performance.
- The next major upgrade should be a dedicated live-odds feed so the displayed
  model edge reflects currently available sportsbook prices.
