
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------
# 1. DASHBOARD CONFIGURATION
# ------------------------------------------------

st.set_page_config(
    page_title="NIFTY Market Analytics",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NIFTY Market Analytics")
st.caption("Interactive analysis of 10 NSE stocks and the NIFTY 50 index")

# ------------------------------------------------
# 2. LOAD CLEANED DATASETS
# ------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "cleaned")

@st.cache_data
def load_data():

    stocks = pd.read_csv(
        os.path.join(DATA_PATH, "all_stocks_cleaned.csv")
    )

    nifty = pd.read_csv(
        os.path.join(DATA_PATH, "nifty50_cleaned.csv")
    )

    # Convert original column names into simple names
    stocks = stocks.rename(columns={
        "Symbol": "symbol",
        "Date": "date",
        "Close Price": "close",
        "Total Traded Quantity": "volume",
        "% Dly Qt to Traded Qty": "delivery_pct",
        "Daily Return (%)": "daily_return"
    })

    nifty = nifty.rename(columns={
        "Date": "date",
        "Close": "close",
        "Daily Return (%)": "daily_return"
    })

    # Convert dates and numerical columns
    stocks["date"] = pd.to_datetime(stocks["date"], errors="coerce")
    nifty["date"] = pd.to_datetime(nifty["date"], errors="coerce")

    for col in ["close", "volume", "delivery_pct", "daily_return"]:
        stocks[col] = pd.to_numeric(stocks[col], errors="coerce")

    for col in ["close", "daily_return"]:
        nifty[col] = pd.to_numeric(nifty[col], errors="coerce")

    stocks = stocks.dropna(subset=["date", "symbol", "close"])
    nifty = nifty.dropna(subset=["date", "close"])

    stocks = stocks.sort_values(["symbol", "date"]).copy()
    nifty = nifty.sort_values("date").copy()

    # Calculate daily returns if the column is missing
    if "daily_return" not in stocks.columns:
        stocks["daily_return"] = (
            stocks.groupby("symbol")["close"].pct_change() * 100
        )

    if "daily_return" not in nifty.columns:
        nifty["daily_return"] = nifty["close"].pct_change() * 100

    return stocks, nifty


try:
    stocks, nifty = load_data()
except Exception as e:
    st.error("Unable to load the datasets. Check your folder paths and CSV columns.")
    st.code(str(e))
    st.stop()

# ------------------------------------------------
# 3. SIDEBAR FILTERS
# ------------------------------------------------

st.sidebar.title("📊 Dashboard Filters")

symbols = sorted(stocks["symbol"].dropna().unique())

selected_stock = st.sidebar.selectbox(
    "Select a Stock",
    symbols
)

min_date = max(stocks["date"].min(), nifty["date"].min()).date()
max_date = min(stocks["date"].max(), nifty["date"].max()).date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) != 2:
    st.warning("Please select both a start date and an end date.")
    st.stop()

start_date, end_date = date_range

if start_date > end_date:
    st.warning("Start date cannot be after end date.")
    st.stop()

# Filter datasets
stock_data = stocks[
    (stocks["date"].dt.date >= start_date) &
    (stocks["date"].dt.date <= end_date)
].copy()

nifty_data = nifty[
    (nifty["date"].dt.date >= start_date) &
    (nifty["date"].dt.date <= end_date)
].copy()

selected_data = stock_data[
    stock_data["symbol"] == selected_stock
].sort_values("date").copy()

if selected_data.empty or nifty_data.empty:
    st.warning("No data available for this selection.")
    st.stop()

# ------------------------------------------------
# 4. CALCULATE ADDITIONAL METRICS
# ------------------------------------------------

# Daily returns
selected_data["return"] = selected_data["close"].pct_change() * 100

# Moving averages
selected_data["MA20"] = selected_data["close"].rolling(20).mean()
selected_data["MA50"] = selected_data["close"].rolling(50).mean()

# Rolling volatility
selected_data["rolling_volatility"] = (
    selected_data["return"].rolling(20).std() * np.sqrt(252)
)

# Absolute daily return for price-volume analysis
selected_data["absolute_return"] = selected_data["return"].abs()

# Annualized volatility for all stocks
volatility = (
    stock_data.groupby("symbol")["daily_return"]
    .std() * np.sqrt(252)
).reset_index(name="annualized_volatility")

# ------------------------------------------------
# 5. HEADER AND KPI CARDS
# ------------------------------------------------

st.subheader(f"Market Overview - {selected_stock}")

stock_return = (
    selected_data["close"].iloc[-1] /
    selected_data["close"].iloc[0] - 1
) * 100

nifty_return = (
    nifty_data["close"].iloc[-1] /
    nifty_data["close"].iloc[0] - 1
) * 100

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric(
    "Latest Closing Price",
    f"₹{selected_data['close'].iloc[-1]:,.2f}"
)

k2.metric(
    "Stock Return",
    f"{stock_return:.2f}%"
)

k3.metric(
    "NIFTY Return",
    f"{nifty_return:.2f}%"
)

k4.metric(
    "Average Trading Volume",
    f"{selected_data['volume'].mean():,.0f}"
)

k5.metric(
    "Average Delivery %",
    f"{selected_data['delivery_pct'].mean():.2f}%"
)

st.divider()

# ------------------------------------------------
# 6. DASHBOARD TABS
# ------------------------------------------------

tabs = st.tabs([
    "🏠 Overview",
    "💹 Price & Returns",
    "📊 Volume",
    "🌊 Volatility",
    "📦 Delivery",
    "🇮🇳 NIFTY 50"
])

# ------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------

with tabs[0]:

    st.header("Project Overview")

    st.write("""
    This project analyzes historical market data of 10 selected
    NSE stocks and the NIFTY 50 index using Python and data
    visualization techniques.
    """)

    c1, c2, c3 = st.columns(3)

    c1.metric("Stocks Analyzed", stocks["symbol"].nunique())
    c2.metric("Trading Days", stock_data["date"].nunique())
    c3.metric(
        "Analysis Period",
        f"{start_date} to {end_date}"
    )

    # Compare returns of all stocks
    performance = (
        stock_data.sort_values("date")
        .groupby("symbol")["close"]
        .agg(lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100)
        .reset_index(name="return")
    )

    fig = px.bar(
        performance,
        x="return",
        y="symbol",
        orientation="h",
        color="return",
        color_continuous_scale="RdYlGn",
        title="Overall Return Comparison of All Stocks",
        labels={"return": "Return (%)", "symbol": "Stock"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"{selected_stock} returned {stock_return:.2f}% "
        f"compared with NIFTY 50 at {nifty_return:.2f}%."
    )

# ------------------------------------------------
# TAB 2: PRICE AND RETURNS - ANALYSES 1, 2, 3
# ------------------------------------------------

with tabs[1]:

    st.header("Price and Returns Analysis")

    # Analysis 1: Historical Price Trend
    st.subheader("1. Historical Stock Price Trend")

    fig = px.line(
        selected_data,
        x="date",
        y=["close", "MA20", "MA50"],
        title=f"{selected_stock} - Price and Moving Averages",
        labels={
            "date": "Date",
            "value": "Price (₹)",
            "variable": "Price Indicator"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    # Analysis 2: Return Distribution
    st.subheader("2. Daily Return Distribution")

    returns = selected_data["return"].dropna()

    fig = px.histogram(
        returns,
        nbins=30,
        title="Distribution of Daily Stock Returns",
        labels={"value": "Daily Return (%)", "count": "Frequency"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write(f"Average daily return: {returns.mean():.3f}%")
    st.write(f"Standard deviation: {returns.std():.3f}%")

    # Analysis 3: Stock vs NIFTY Performance
    st.subheader("3. Stock vs NIFTY 50 Performance")

    comparison = pd.merge(
        selected_data[["date", "close"]],
        nifty_data[["date", "close"]],
        on="date",
        suffixes=("_stock", "_nifty")
    )

    comparison["Stock Base 100"] = (
        comparison["close_stock"] /
        comparison["close_stock"].iloc[0] * 100
    )

    comparison["NIFTY Base 100"] = (
        comparison["close_nifty"] /
        comparison["close_nifty"].iloc[0] * 100
    )

    fig = px.line(
        comparison,
        x="date",
        y=["Stock Base 100", "NIFTY Base 100"],
        title="Stock Performance Compared with NIFTY 50",
        labels={"value": "Indexed Performance", "date": "Date"}
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# TAB 3: VOLUME - ANALYSES 4 AND 5
# ------------------------------------------------

with tabs[2]:

    st.header("Trading Volume Analysis")

    # Analysis 4: Trading Volume Pattern
    st.subheader("4. Trading Volume Pattern")

    fig = px.bar(
        selected_data,
        x="date",
        y="volume",
        title=f"{selected_stock} - Daily Trading Volume",
        labels={"date": "Date", "volume": "Traded Quantity"}
    )

    st.plotly_chart(fig, use_container_width=True)

    # Analysis 5: Price-Volume Relationship
    st.subheader("5. Price-Volume Relationship")

    volume_data = selected_data.dropna(
        subset=["volume", "absolute_return"]
    )

    corr_pv = volume_data["volume"].corr(
        volume_data["absolute_return"]
    )

    fig = px.scatter(
        volume_data,
        x="volume",
        y="absolute_return",
        title="Trading Volume vs Magnitude of Daily Price Movement",
        labels={
            "volume": "Trading Volume",
            "absolute_return": "Absolute Daily Return (%)"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"Correlation between trading volume and absolute daily "
        f"return: {corr_pv:.3f}"
    )

# ------------------------------------------------
# TAB 4: VOLATILITY - ANALYSIS 6
# ------------------------------------------------

with tabs[3]:

    st.header("Stock Volatility Analysis")

    # Analysis 6: Volatility Comparison
    st.subheader("6. Annualized Volatility Comparison")

    fig = px.bar(
        volatility,
        x="symbol",
        y="annualized_volatility",
        color="annualized_volatility",
        title="Risk Comparison of All 10 Stocks",
        labels={
            "symbol": "Stock",
            "annualized_volatility": "Annualized Volatility (%)"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"{selected_stock} - Rolling Volatility")

    fig = px.line(
        selected_data,
        x="date",
        y="rolling_volatility",
        title="20-Day Rolling Annualized Volatility",
        labels={
            "date": "Date",
            "rolling_volatility": "Volatility (%)"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# TAB 5: DELIVERY - ANALYSES 7 AND 8
# ------------------------------------------------

with tabs[4]:

    st.header("Delivery Analysis")

    # Analysis 7: Delivery Percentage
    st.subheader("7. Delivery Percentage Analysis")

    avg_delivery = (
        stock_data.groupby("symbol")["delivery_pct"]
        .mean()
        .reset_index(name="average_delivery")
    )

    fig = px.bar(
        avg_delivery,
        x="symbol",
        y="average_delivery",
        title="Average Delivery Percentage of All Stocks",
        labels={
            "symbol": "Stock",
            "average_delivery": "Delivery (%)"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    fig = px.line(
        selected_data,
        x="date",
        y="delivery_pct",
        title=f"{selected_stock} - Daily Delivery Percentage",
        labels={"date": "Date", "delivery_pct": "Delivery (%)"}
    )

    st.plotly_chart(fig, use_container_width=True)

    # Analysis 8: Delivery vs Price Movement
    st.subheader("8. Delivery vs Price Movement")

    delivery_data = selected_data.dropna(
        subset=["delivery_pct", "return"]
    )

    corr_dv = delivery_data["delivery_pct"].corr(
        delivery_data["return"]
    )

    fig = px.scatter(
        delivery_data,
        x="delivery_pct",
        y="return",
        title="Delivery Percentage vs Daily Stock Return",
        labels={
            "delivery_pct": "Delivery Percentage (%)",
            "return": "Daily Return (%)"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(f"Delivery vs price return correlation: {corr_dv:.3f}")

# ------------------------------------------------
# TAB 6: NIFTY 50 - ANALYSES 9 AND 10
# ------------------------------------------------

with tabs[5]:

    st.header("NIFTY 50 Index Analysis")

    n = nifty_data.sort_values("date").copy()

    # Analysis 9: NIFTY Index Performance
    st.subheader("9. NIFTY 50 Performance")

    n["daily_return"] = n["close"].pct_change() * 100

    n["cumulative_return"] = (
        (1 + n["daily_return"].fillna(0) / 100).cumprod() * 100
    )

    n["drawdown"] = (
        n["close"] / n["close"].cummax() - 1
    ) * 100

    fig = px.line(
        n,
        x="date",
        y="close",
        title="NIFTY 50 Historical Closing Price",
        labels={"date": "Date", "close": "Closing Price"}
    )

    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "NIFTY Total Return",
            f"{nifty_return:.2f}%"
        )

    with c2:
        st.metric(
            "Maximum Drawdown",
            f"{n['drawdown'].min():.2f}%"
        )

    # Analysis 10: Volatility vs Index Movement
    st.subheader("10. Volatility vs Index Movement")

    n["absolute_return"] = n["daily_return"].abs()

    n["rolling_volatility"] = (
        n["daily_return"].rolling(20).std() * np.sqrt(252)
    )

    corr_vi = n["absolute_return"].corr(
        n["rolling_volatility"]
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=n["date"],
        y=n["absolute_return"],
        name="Absolute Daily Return",
        mode="lines"
    ))

    fig.add_trace(go.Scatter(
        x=n["date"],
        y=n["rolling_volatility"],
        name="20-Day Annualized Volatility",
        mode="lines",
        yaxis="y2"
    ))

    fig.update_layout(
        title="NIFTY Movement vs Rolling Volatility",
        xaxis_title="Date",
        yaxis_title="Absolute Daily Return (%)",
        yaxis2=dict(
            title="Annualized Volatility (%)",
            overlaying="y",
            side="right"
        ),
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"Correlation between absolute NIFTY daily returns and "
        f"rolling volatility: {corr_vi:.3f}"
    )

# ------------------------------------------------
# 7. FOOTER
# ------------------------------------------------

st.divider()

st.caption(
    "NIFTY Market Analytics | Python | Pandas | NumPy | Plotly | Streamlit"
)
