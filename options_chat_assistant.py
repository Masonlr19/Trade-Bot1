import streamlit as st
import pandas as pd
import requests
from ta import add_all_ta_features
from ta.utils import dropna
from newsapi import NewsApiClient
from textblob import TextBlob

# Initialize NewsAPI client (replace with your own key)
newsapi = NewsApiClient(api_key='your_newsapi_key_here')

# Your Alpha Vantage API key here
ALPHAVANTAGE_API_KEY = 'your_alpha_vantage_api_key_here'

# Fetch stock data from Alpha Vantage
def fetch_stock_data(symbol, outputsize="compact"):
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": outputsize,  # "compact" (last 100 days) or "full"
        "apikey": ALPHAVANTAGE_API_KEY,
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Check for error message
    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
    if "Time Series (Daily)" not in data:
        raise ValueError("Unexpected response format from Alpha Vantage.")

    ts = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(ts, orient="index")
    df = df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. adjusted close": "Adj Close",
        "6. volume": "Volume",
        # other columns are available but not used here
    })

    # Convert columns to numeric
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Sort by date ascending
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Drop rows with missing required columns
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
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
st.title("📊 Options Chat Assistant with Alpha Vantage")
symbol = st.text_input("Enter a stock symbol (e.g., AAPL):", value="AAPL")

if symbol:
    with st.spinner("Fetching data..."):
        try:
            df = fetch_stock_data(symbol)
            df = analyze_data(df)
        except Exception as e:
            st.error(f"Error fetching or analyzing data: {e}")
            st.stop()

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

