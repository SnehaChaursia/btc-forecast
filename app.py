import streamlit as st
import numpy as np
import pandas as pd
import requests
from arch import arch_model
import plotly.graph_objects as go
import json, os

st.set_page_config(page_title="BTC Forecast", layout="wide")

LOOKBACK = 500
N_SIMS = 3000

@st.cache_data(ttl=300)
def fetch_btc(limit=600):
    url = "https://data-api.binance.vision/api/v3/klines"
    r = requests.get(url, params={"symbol":"BTCUSDT","interval":"1h","limit":limit}, timeout=10)
    df = pd.DataFrame(r.json(), columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","tb","tq","ig"
    ])
    df["close"] = df["close"].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    return df["close"]

def make_prediction(prices):
    log_ret = np.log(prices / prices.shift(1)).dropna() * 100
    am = arch_model(log_ret, vol='Garch', p=1, q=1, dist='studentst')
    res = am.fit(disp='off', show_warning=False)
    sigma_h = res.conditional_volatility.iloc[-1] / 100
    nu = max(4.0, float(res.params.get('nu', 6.0)))
    S0 = float(prices.iloc[-1])
    mu = log_ret.mean() / 100
    Z = np.random.standard_t(nu, size=N_SIMS) * np.sqrt((nu - 2) / nu)
    ST = S0 * np.exp((mu - 0.5 * sigma_h**2) + sigma_h * Z)
    return float(np.percentile(ST, 2.5)), float(np.percentile(ST, 97.5)), S0

# Load backtest metrics if file exists
cov, winkler_mean, avg_width = None, None, None
if os.path.exists("backtest_results.jsonl"):
    bt = [json.loads(l) for l in open("backtest_results.jsonl")]
    first_keys = list(bt[0].keys())
    cov_key     = [k for k in first_keys if "coverage" in k.lower()][0]
    winkler_key = [k for k in first_keys if "winkler" in k.lower()][0]
    width_key   = [k for k in first_keys if "width" in k.lower()][0]
    cov          = np.mean([r[cov_key]     for r in bt])
    winkler_mean = np.mean([r[winkler_key] for r in bt])
    avg_width    = np.mean([r[width_key]   for r in bt])

# Fetch data and predict
prices = fetch_btc()
low95, high95, S0 = make_prediction(prices.iloc[-LOOKBACK:])

# Header
st.title("BTC/USDT — Next Hour Forecast")
st.caption("Model: GARCH(1,1) + Student-t | Data: Binance public API")

# Metrics row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current BTC price", f"${S0:,.0f}")
c2.metric("95% low (next hour)", f"${low95:,.0f}")
c3.metric("95% high (next hour)", f"${high95:,.0f}")
c4.metric("Predicted range", f"${high95-low95:,.0f}")
if cov is not None:
    c5.metric("Backtest coverage", f"{cov:.1%}")

# Backtest metrics row
if winkler_mean is not None:
    st.info(f"Backtest metrics — Coverage: {cov:.1%} | Avg width: ${avg_width:,.0f} | Mean Winkler: {winkler_mean:,.0f}")

st.divider()

# Chart: last 50 bars + predicted range for next bar
last50 = prices.iloc[-50:]
next_t = last50.index[-1] + pd.Timedelta(hours=1)

fig = go.Figure()

# Price line
fig.add_trace(go.Scatter(
    x=last50.index, y=last50.values,
    name="BTC price",
    line=dict(color="#378ADD", width=2)
))

# Shaded prediction ribbon
fig.add_trace(go.Scatter(
    x=[last50.index[-1], next_t, next_t, last50.index[-1]],
    y=[high95, high95, low95, low95],
    fill='toself',
    fillcolor='rgba(29,158,117,0.15)',
    line=dict(color='rgba(29,158,117,0.4)'),
    name='95% predicted range'
))

# Dotted lines for low and high
fig.add_hline(y=low95,  line_dash="dot", line_color="#E24B4A",
              annotation_text=f"Low ${low95:,.0f}", annotation_position="right")
fig.add_hline(y=high95, line_dash="dot", line_color="#1D9E75",
              annotation_text=f"High ${high95:,.0f}", annotation_position="right")

fig.update_layout(
    title="Last 50 hours of BTC price + next-hour predicted range",
    xaxis_title="Time (UTC)",
    yaxis_title="Price (USDT)",
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)

st.plotly_chart(fig, use_container_width=True)
st.caption("Shaded green area = 95% confidence range for the next hourly bar. Refreshes every 5 minutes.")
