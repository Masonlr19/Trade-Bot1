import streamlit as st
import yfinance as yf
import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna
from newsapi import NewsApiClient
import subprocess
import sys

# Ensure textblob is installed and corpora are downloaded
try:
    from textblob import TextBlob
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "textblob"])
    from textblob import TextBlob
    import nltk
    nltk.download('brown')
    nltk.download('punkt')

# Initialize NewsAPI client (replace with your own key)
newsapi = NewsApiClient(api_key='your_newsapi_key_here')

# Fetch stock data
def fetch_stock_data(symbol, period="6mo", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval)
    required_cols = {"Open", "High", "Low", "Close", "Volume"}

    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

    df = df.dropna(subset=required_cols)
    return df

# Analyze data with TA indicators
def analyze_data(df):
    try:
        df = dropna(df)
        df = add_all_ta_features(
            df,
            open="Open", high="High", low="Low", close="Close", volume="Volume",
            fillna=True
        )
        return df
    except Exception as e:
        st.error(f"TA-lib error: {e}")
        st.stop()

# Fetch latest news for a symbol
def fetch_news(symbol):
    try:
        query = f"{symbol} stock"
        articles = newsapi.get_everything(q=query, language="en", sort_by="publishedAt", page_size=5)
        return articles['articles']
    except Exception as e:
        st.error(f"NewsAPI error: {e}")
        return []

# Analyze news sentiment
def analyze_news_sentiment(articles):
    polarity = 0
    for article in articles:
        if article.get("description"):
            blob = TextBlob(article["description"])
            polarity += blob.sentiment.polarity
    avg_polarity = polarity / len(articles) if articles else 0
    return avg_polarity

# Generate trade recommendation
def generate_trade_signal(df, sentiment_score):
    latest = df.iloc[-1]
    rsi = latest.get("momentum_rsi")
    macd = latest.get("trend_macd")
    signal = "Hold"

    if rsi is not None:
        if rsi < 30:
            signal = "Buy"
        elif rsi > 70:
            signal = "Sell"

    if sentiment_score > 0.3 and signal == "Hold":
        signal = "Buy"
    elif sentiment_score < -0.3 and signal == "Hold":
        signal = "Sell"

    return signal

# Streamlit UI
st.title("📊 Options Chat Assistant")
symbol = st.text_input("Enter a stock symbol (e.g., AAPL):", value="AAPL")

if symbol:
    with st.spinner("Fetching data..."):
        df = fetch_stock_data(symbol)
        df = analyze_data(df)

    st.subheader("Technical Analysis Summary")
    st.write(df.tail(1).T)

    st.subheader("Latest News")
    news_items = fetch_news(symbol)
    for article in news_items:
        st.markdown(f"### [{article['title']}]({article['url']})")
        st.write(article['description'])

    sentiment_score = analyze_news_sentiment(news_items)
    st.write(f"🧠 News Sentiment Score: `{sentiment_score:.2f}`")

    st.subheader("📈 Trade Recommendation")
    signal = generate_trade_signal(df, sentiment_score)
    st.success(f"Recommended Action: **{signal}**")
