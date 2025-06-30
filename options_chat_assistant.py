import os
import streamlit as st
import yfinance as yf
import pandas as pd
import openai
from ta import add_all_ta_features
from ta.utils import dropna
import requests

# 📌 API Key (set this in Streamlit as a secret or env var)
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# 🔍 Fetch stock data
def fetch_stock_data(symbol, period="6mo", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval)
    df = df.dropna()
    return df

# 📈 Analyze with technical indicators
def analyze_data(df):
    df = dropna(df)
    df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")
    return df

# 📰 Fetch news from NewsAPI
def fetch_news(symbol):
    if not NEWS_API_KEY:
        return ["NEWS_API_KEY is not set."]
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={symbol}&"
        f"sortBy=publishedAt&"
        f"language=en&"
        f"apiKey={NEWS_API_KEY}"
    )
    try:
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get("articles", [])[:5]
        return [f"{a['title']} - {a['source']['name']}" for a in articles]
    except Exception as e:
        return [f"Error fetching news: {e}"]

# 🤖 ChatGPT Prompt Helper
def summarize_data(df, news):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    description = df.tail(1).to_string()
    news_str = "\n".join(news)

    prompt = (
        f"Here is the latest stock technical data:\n{description}\n\n"
        f"Recent news headlines:\n{news_str}\n\n"
        "Give a concise summary with any bullish or bearish insights."
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response['choices'][0]['message']['content']

# 🌐 Streamlit UI
st.title("📊 Options Chat Assistant")

symbol = st.text_input("Enter stock symbol (e.g., AAPL)", "AAPL")

if symbol:
    with st.spinner("Fetching data..."):
        df = fetch_stock_data(symbol)
        df = analyze_data(df)
        news = fetch_news(symbol)
        summary = summarize_data(df, news)

        st.subheader("📈 Technical Summary")
        st.write(df.tail(1))

        st.subheader("📰 Recent News")
        for article in news:
            st.markdown(f"- {article}")

        st.subheader("🧠 AI Summary")
        st.write(summary)
