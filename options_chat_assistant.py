# ============ financial_advisor_bot.py ============
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import requests
import time
from transformers import pipeline
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ============ SETUP ============
st.set_page_config(page_title="AI Financial Advisor", layout="wide")
st.title("📊 AI Financial Advisor Bot")
st.markdown("Combining market data, options analysis, LLMs, and backtesting.")

# ============ NEWS SENTIMENT ============
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_sentiment_pipeline()

def analyze_news_sentiment(news_titles):
    results = sentiment_pipeline(news_titles)
    return results

# ============ MARKET + OPTIONS DATA ============
def get_stock_data(ticker, period="6mo", interval="1d", retries=3, delay=2):
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            return stock.history(period=period, interval=interval)
        except Exception as e:
            st.warning(f"Attempt {attempt+1} to fetch stock data failed: {e}")
            time.sleep(delay)
    st.error("Failed to retrieve stock data after multiple attempts.")
    return pd.DataFrame()

def get_news_titles(ticker):
    # Placeholder for real news API
    return [
        f"{ticker} beats quarterly earnings expectations",
        f"{ticker} stock downgraded by analysts",
        f"{ticker} shows strong growth potential in AI sector"
    ]

def get_options_data(ticker, retries=3, delay=2):
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            expiry = stock.options[0]
            options_chain = stock.option_chain(expiry)
            return options_chain.calls[['strike', 'lastPrice', 'volume', 'impliedVolatility']].head()
        except Exception as e:
            st.warning(f"Attempt {attempt+1} to fetch options data failed: {e}")
            time.sleep(delay)
    st.error("Failed to retrieve options data after multiple attempts.")
    return pd.DataFrame()

# ============ BACKTESTING ============
def backtest_strategy(prices):
    daily_returns = prices['Close'].pct_change().dropna()
    cumulative = (1 + daily_returns).cumprod()
    return cumulative

# ============ TRADE RECOMMENDATION ============
def recommend_trade(stock_data, sentiments, options_df):
    last_price = stock_data['Close'].iloc[-1]
    trend = stock_data['Close'].pct_change().rolling(5).mean().iloc[-1]
    sentiment_score = sum(1 if s['label'] == 'POSITIVE' else -1 for s in sentiments)
    sentiment_text = ', '.join([s['label'] for s in sentiments])

    recommendation = "HOLD"
    reason = []

    if trend > 0.01:
        reason.append("Upward trend in stock price.")
        if sentiment_score > 0:
            recommendation = "BUY CALL OPTIONS"
            reason.append("Positive sentiment from news articles.")
    elif trend < -0.01:
        reason.append("Downward trend in stock price.")
        if sentiment_score < 0:
            recommendation = "BUY PUT OPTIONS"
            reason.append("Negative sentiment from news articles.")

    if options_df.empty:
        reason.append("Options data unavailable.")
    else:
        reason.append("Options chain shows available liquidity.")

    return recommendation, reason, sentiment_text

# ============ DAILY MOCK MODEL ============
@st.cache_data
def train_mock_model(data):
    X = np.array(data['Close'].pct_change().dropna()).reshape(-1, 1)
    y = (X > 0).astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_scaled, y.ravel())
    return model, scaler

# ============ UI INTERACTION ============
ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA):", "AAPL")

if st.button("Run Analysis"):
    with st.spinner("Fetching data and analyzing..."):
        stock_data = get_stock_data(ticker)
        options_data = get_options_data(ticker)
        news_titles = get_news_titles(ticker)
        sentiments = analyze_news_sentiment(news_titles)

        recommendation, reasons, sentiment_summary = recommend_trade(stock_data, sentiments, options_data)
        model, scaler = train_mock_model(stock_data)

        st.subheader(f"{ticker} Price History")
        st.line_chart(stock_data['Close'])

        st.subheader("Options Chain (Calls - Near Expiry)")
        st.dataframe(options_data)

        st.subheader("🔍 Trade Recommendation")
        st.markdown(f"**Recommendation:** {recommendation}")
        st.markdown(f"**Sentiment Summary:** {sentiment_summary}")
        st.markdown("**Explanation:**")
        for r in reasons:
            st.markdown(f"- {r}")

        st.subheader("📈 Backtest Simple Buy-Hold Strategy")
        backtest = backtest_strategy(stock_data)
        st.line_chart(backtest)

        st.subheader("💼 Simulated Portfolio Performance")
        shares = st.number_input("Number of Shares Owned:", 0, 10000, 10)
        value = shares * stock_data['Close'].iloc[-1]
        st.markdown(f"**Current Portfolio Value:** ${value:,.2f}")

        st.success("Analysis complete.")
