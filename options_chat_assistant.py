import streamlit as st
import yfinance as yf
import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna
from newsapi import NewsApiClient

# Initialize NewsAPI client (replace with your own key)
newsapi = NewsApiClient(api_key='your_newsapi_key_here')

# Fetch stock data
def fetch_stock_data(symbol, period="6mo", interval="1d"):    
    df = yf.download(symbol, period=period, interval=interval)
    
    if df.empty:
        st.error(f"No data found for symbol: {symbol}")
        st.stop()

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)

    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

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
