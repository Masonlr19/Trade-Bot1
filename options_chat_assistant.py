# financial_advisor_bot.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import time
import os
from transformers import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# --- OpenAI ---
import openai
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

def ask_openai(prompt):
    if client is None:
        return "OpenAI client is not initialized. Please set the OPENAI_API_KEY environment variable."
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a Wall Street-level financial advisor."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI error: {e}"


        
st.set_page_config(page_title="AI Financial Advisor", layout="wide")
st.title("📊 AI Financial Advisor Bot")
st.markdown("Combining market data, options analysis, LLMs, and backtesting.")

# --- Tradier config ---
TRADIER_TOKEN = os.getenv("TRADIER_TOKEN")
if not TRADIER_TOKEN:
    st.error("Please set the TRADIER_TOKEN environment variable!")
HEADERS = {
    "Authorization": f"Bearer {TRADIER_TOKEN}",
    "Accept": "application/json"
}

# --- Retry decorator ---
def retry_request(func, retries=3, delay=2, backoff=2, *args, **kwargs):
    current_delay = delay
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(current_delay)
                current_delay *= backoff
            else:
                raise e

# --- Tradier API calls ---

@st.cache_data(ttl=600)
def get_stock_history(symbol, period_days=180):
    end = datetime.date.today()
    start = end - datetime.timedelta(days=period_days)
    url = "https://api.tradier.com/v1/markets/history"
    params = {
        "symbol": symbol,
        "interval": "daily",
        "start": start.isoformat(),
        "end": end.isoformat()
    }
    def request():
        resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    data = retry_request(request)
    if "history" not in data or "day" not in data["history"]:
        return pd.DataFrame()
    days = data["history"]["day"]
    df = pd.DataFrame(days)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    # Convert columns to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

@st.cache_data(ttl=300)
def get_stock_quote(symbol):
    url = "https://api.tradier.com/v1/markets/quotes"
    params = {"symbols": symbol}
    def request():
        resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    data = retry_request(request)
    try:
        quote = data["quotes"]["quote"]
        # If multiple quotes returned, take first
        if isinstance(quote, list):
            quote = quote[0]
        last_price = float(quote["last"])
        return last_price
    except Exception:
        return None

@st.cache_data(ttl=600)
def get_options_chain(symbol, expiration=None):
    # Step 1: Get expirations
    url_exp = "https://api.tradier.com/v1/markets/options/expirations"
    params_exp = {"symbol": symbol}
    def request_exp():
        resp = requests.get(url_exp, headers=HEADERS, params=params_exp, timeout=10)
        resp.raise_for_status()
        return resp.json()
    data_exp = retry_request(request_exp)
    expirations = data_exp.get("expirations", {}).get("date", [])
    if not expirations:
        return pd.DataFrame()
    expiry = expiration or expirations[0]

    # Step 2: Get option chains for that expiration
    url_chain = "https://api.tradier.com/v1/markets/options/chains"
    params_chain = {
        "symbol": symbol,
        "expiration": expiry,
        "greeks": "true"
    }
    def request_chain():
        resp = requests.get(url_chain, headers=HEADERS, params=params_chain, timeout=10)
        resp.raise_for_status()
        return resp.json()
    data_chain = retry_request(request_chain)
    options = data_chain.get("options", {}).get("option", [])
    if not options:
        return pd.DataFrame()

    df = pd.DataFrame(options)

    # Ensure numeric columns
    for col in ['strike', 'bid', 'ask', 'last', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

# --- News Sentiment ---

@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_sentiment_pipeline()

def analyze_news_sentiment(news_titles):
    return sentiment_pipeline(news_titles)

def get_news_titles(ticker):
    # Placeholder: Replace with real news API integration later
    return [
        f"{ticker} beats quarterly earnings expectations",
        f"{ticker} stock downgraded by analysts",
        f"{ticker} shows strong growth potential in AI sector"
    ]

# --- Backtesting ---

def backtest_strategy(prices):
    daily_returns = prices['close'].pct_change().dropna()
    cumulative = (1 + daily_returns).cumprod()
    return cumulative

# --- Recommendation logic ---

def recommend_trade(stock_data, sentiments, options_df):
    if stock_data.empty:
        return "NO DATA", ["Stock data unavailable."], ""

    last_price = stock_data['close'].iloc[-1]
    trend = stock_data['close'].pct_change().rolling(5).mean().iloc[-1]
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

# --- Mock model training ---

@st.cache_data
def train_mock_model(data):
    if data.empty:
        return None, None
    X = np.array(data['close'].pct_change().dropna()).reshape(-1, 1)
    y = (X > 0).astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_scaled, y.ravel())
    return model, scaler

# --- UI ---

ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA):", "AAPL")

if st.button("Run Analysis"):
    if not TRADIER_TOKEN:
        st.error("No Tradier API token configured! Set TRADIER_TOKEN environment variable.")
    else:
        with st.spinner("Fetching data and analyzing..."):
            try:
                stock_data = get_stock_history(ticker)
                options_data = get_options_chain(ticker)
                news_titles = get_news_titles(ticker)
                sentiments = analyze_news_sentiment(news_titles)

                recommendation, reasons, sentiment_summary = recommend_trade(stock_data, sentiments, options_data)
                model, scaler = train_mock_model(stock_data)

                st.subheader(f"{ticker} Price History")
                st.line_chart(stock_data['close'])

                st.subheader("Options Chain")
                st.dataframe(options_data[['strike', 'bid', 'ask', 'last', 'volume', 'implied_volatility']] if not options_data.empty else pd.DataFrame())

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
                value = shares * stock_data['close'].iloc[-1] if not stock_data.empty else 0
                st.markdown(f"**Current Portfolio Value:** ${value:,.2f}")

                st.success("Analysis complete.")
            except Exception as e:
                st.error(f"Error during analysis: {e}")

# --- Ask the AI Assistant ---
st.markdown("---")
st.subheader("🧠 Ask the AI Financial Advisor")

user_question = st.text_input("Enter any financial question or request an explanation:")
if st.button("Get AI Answer"):
    if not OPENAI_API_KEY:
        st.error("OpenAI API key not configured.")
    elif not user_question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            ai_response = ask_openai(user_question)
            st.markdown(f"**Answer:**\n\n{ai_response}")
