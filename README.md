# BTC/USDT — Next Hour Forecast

A live Bitcoin price range forecaster built for the AlphaI × Polaris Build Challenge.

## What it does

Every hour, a new BTC candle closes. This app predicts the price range where BTC will land in the **next hour** with 95% confidence.

Example: "I'm 95% sure BTC will be between $78,089 and $78,889 one hour from now."

## Live Dashboard

[Open the live dashboard](https://btc-forecast-xbfzhssrmngj9iu7hwgs.streamlit.app)

The dashboard shows:
- Current BTC price (live from Binance)
- Predicted 95% range for the next hour
- Chart of last 50 hourly bars with shaded prediction ribbon
- Backtest metrics (coverage, avg width, Winkler score)

## Backtest Results (Part A)

Evaluated on 500 hourly bars of BTCUSDT data:

| Metric | Result | Target |
|---|---|---|
| Coverage 95% | 95.8% | ~95% |
| Avg width | $1,375 | narrower = better |
| Mean Winkler | 1,719 | lower = better |

Coverage of 95.8% means the model captured the actual next-hour price inside its predicted range 95.8% of the time — nearly exactly the target.

## How it works

**Data** — Fetches the last 500 hourly BTCUSDT bars from Binance's public API (no API key needed).

**Model** — GARCH(1,1) with Student-t distribution:
- GARCH captures volatility clustering — calm hours cluster together, volatile hours cluster together
- Student-t handles fat tails — BTC has frequent large moves that a normal distribution would miss
- Monte Carlo simulation — runs 3,000 simulations of the next hour and reads off the 2.5th and 97.5th percentiles as the 95% range

**No-peeking** — The backtest is strict: when predicting bar N, only bars 0 to N-1 are used. No future data leaks into any prediction.

## Bugs fixed from starter notebook

- Starter used `FIGARCH` which is very slow for hourly data — replaced with `GARCH(1,1)`
- Starter `train=504, test=252` assumed daily data — changed to `train=200, test=500` to fit 720 hourly bars
- Removed broken `add_vline` datetime call in Plotly
- Fixed `symbol` variable missing after data source swap
- Fixed JSON serialization error for numpy types in backtest results

## Project structure
```
btc-forecast/
├── app.py                  # Streamlit dashboard
├── backtest_results.jsonl  # 500 predictions with actuals
├── requirements.txt        # Python dependencies
└── README.md

```
## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dependencies

- `streamlit` — dashboard framework
- `arch` — GARCH model
- `plotly` — interactive charts
- `pandas` / `numpy` / `scipy` — data processing
- `requests` — Binance API calls
