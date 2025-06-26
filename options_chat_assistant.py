import streamlit as st
import yfinance
import plotly.graph_objs as go
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.news import News
from ta import add_all_ta_features
import pandas as pd
import openai
import os

# Set your API keys
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY") or "YOUR_ALPHA_VANTAGE_KEY"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "YOUR_OPENAI_KEY"
openai.api_key = OPENAI_API_KEY

# Alpha Vantage clients
ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')
news_client = News(key=ALPHA_VANTAGE_KEY)

# Streamlit UI
st.title("📊 Options Trading Assistant")
ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", value="AAPL").upper()

def fetch_stock_data(symbol):
    try:
        df = yf.download(symbol, period="3mo", interval="1d")
        return df
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

def analyze_data(df):
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    existing_cols = [col for col in required_cols if col in df.columns]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        return {
            "RSI": f"Missing columns: {', '.join(missing_cols)}",
            "MACD": "N/A",
            "ADX": "N/A"
        }

    df = df.dropna(subset=existing_cols)
    df = df[df["Volume"] > 0]

    if df.empty or len(df) < 10:
        return {
            "RSI": "Not enough data",
            "MACD": "N/A",
            "ADX": "N/A"
        }

    df = add_all_ta_features(
        df,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume"
    )

    last_row = df.iloc[-1]
    return {
        "RSI": round(last_row.get("momentum_rsi", 0), 2),
        "MACD": round(last_row.get("trend_macd", 0), 2),
        "ADX": round(last_row.get("trend_adx", 0), 2),
    }

def fetch_news(symbol):
    try:
        news_data = news_client.get_news(tickers=[symbol], topics=['financial_markets'], time_from='20240601T0000')
        return news_data.get("feed", [])[:5]
    except Exception as e:
        st.warning(f"News fetch failed: {e}")
        return []

def generate_trade_recommendation(symbol, ta_summary, news_headlines):
    prompt = f"""
You are a financial trading assistant. Given the following:
- Stock symbol: {symbol}
- Technical Indicators: RSI={ta_summary['RSI']}, MACD={ta_summary['MACD']}, ADX={ta_summary['ADX']}
- Recent News Headlines: {news_headlines}

Provide a simple recommendation on whether to BUY, SELL, or HOLD the stock and why.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()

# Run analysis
if ticker:
    df = fetch_stock_data(ticker)

    if df is not None:
        st.subheader(f"{ticker} Price Chart")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candlestick"
        ))
        st.plotly_chart(fig)

        st.subheader("📈 Technical Analysis")
        ta_summary = analyze_data(df)
        st.write(ta_summary)

        st.subheader("📰 Latest News")
        news = fetch_news(ticker)
        headlines = [item.get("title", "No title") for item in news]
        for title in headlines:
            st.markdown(f"- {title}")

        st.subheader("🤖 Trade Recommendation")
        recommendation = generate_trade_recommendation(ticker, ta_summary, headlines)
        st.write(recommendation)
