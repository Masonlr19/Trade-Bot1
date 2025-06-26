import streamlit as st
import yfinance as yf
import requests
import openai
import pandas as pd
import ta
import datetime

# --- CONFIG ---
ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"
openai.api_key = "YOUR_OPENAI_API_KEY"

# --- Alpha Vantage News Fetcher ---
def get_alpha_vantage_news(symbol):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("feed", [])
    return []

# --- Load stock data ---
def load_data(symbol):
    end = datetime.date.today()
    start = end - datetime.timedelta(days=60)
    df = yf.download(symbol, start=start, end=end)
    return df

# --- Perform technical analysis ---
def analyze_data(df):
    df.ta = ta.add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume"
    )
    last_row = df.iloc[-1]
    summary = {
        "RSI": last_row.get("momentum_rsi", "N/A"),
        "MACD": last_row.get("trend_macd", "N/A"),
        "ADX": last_row.get("trend_adx", "N/A"),
    }
    return summary

# --- Generate trade recommendation ---
def get_trade_recommendation(symbol, ta_summary, news_items):
    news_text = "\n\n".join(
        [f"- {item['title']} ({item['summary']})" for item in news_items[:3]]
    )
    prompt = f"""
You are a trading assistant. Based on the following data, provide a trading recommendation for {symbol}.

Technical Summary:
RSI: {ta_summary['RSI']}
MACD: {ta_summary['MACD']}
ADX: {ta_summary['ADX']}

Recent News:
{news_text}

What trade would you suggest (buy, hold, sell) and why?
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message["content"]

# --- UI ---
st.title("📈 AI Options Trading Assistant")

symbol = st.text_input("Enter a stock symbol (e.g. AAPL, MSFT)", "AAPL").upper()

if st.button("Analyze and Recommend"):
    with st.spinner("Loading data and analyzing..."):
        df = load_data(symbol)
        if df.empty:
            st.error("Could not fetch data. Check the symbol.")
        else:
            ta_summary = analyze_data(df)
            news_items = get_alpha_vantage_news(symbol)
            recommendation = get_trade_recommendation(symbol, ta_summary, news_items)

            st.subheader("📊 Technical Indicators")
            st.write(ta_summary)

            st.subheader("📰 Recent News")
            for item in news_items[:3]:
                st.markdown(f"**{item['title']}**  \n{item['summary']}")

            st.subheader("💡 AI Trade Suggestion")
            st.success(recommendation)
