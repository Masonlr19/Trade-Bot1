import streamlit as st
import pandas as pd
import requests
from ta import add_all_ta_features
from ta.utils import dropna
from newsapi import NewsApiClient
import subprocess
import sys
import datetime
import time

# 1. Ensure textblob and nltk data are installed
try:
    from textblob import TextBlob
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "textblob"])
    from textblob import TextBlob
    import nltk
    nltk.download('brown')
    nltk.download('punkt')

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

newsapi = NewsApiClient(api_key='your_newsapi_key_here')
ALPHAVANTAGE_API_KEY = "your_alpha_vantage_api_key_here"

def fetch_stock_data(symbol, outputsize="compact"):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": ALPHAVANTAGE_API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    st.write("Raw API response:", data)

    if "Information" in data:
        raise ValueError(
            f"Alpha Vantage returned an informational message: {data['Information']} This typically means you've hit a rate limit or tried accessing a premium feature."
        )
    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
    if "Note" in data:
        raise ValueError(f"Alpha Vantage API rate limit exceeded: {data['Note']}")
    if "Time Series (Daily)" not in data:
        raise ValueError(f"Unexpected response keys: {list(data.keys())}")

    ts = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(ts, orient="index")
    df = df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. adjusted close": "Adj Close",
        "6. volume": "Volume",
        "7. dividend amount": "Dividend",
        "8. split coefficient": "Split Coef"
    })
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    df = df.dropna(subset=required_cols)
    return df

def fetch_stock_data_weekly(symbol):
    time.sleep(15)  # Prevent hitting the rate limit
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_WEEKLY_ADJUSTED",
        "symbol": symbol,
        "apikey": ALPHAVANTAGE_API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()

    if "Information" in data:
        raise ValueError(
            f"Alpha Vantage returned an informational message: {data['Information']} This typically means you've hit a rate limit or tried accessing a premium feature."
        )
    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
    if "Weekly Adjusted Time Series" not in data:
        raise ValueError(f"Unexpected response keys: {list(data.keys())}")

    ts = data["Weekly Adjusted Time Series"]
    df = pd.DataFrame.from_dict(ts, orient="index")
    df = df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. adjusted close": "Adj Close",
        "6. volume": "Volume",
        "7. dividend amount": "Dividend"
    })
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    df = df.dropna(subset=required_cols)
    return df

def safe_fetch(fetch_func, *args, retries=3, **kwargs):
    for attempt in range(retries):
        try:
            return fetch_func(*args, **kwargs)
        except ValueError as e:
            if "rate limit" in str(e).lower():
                st.warning("Rate limit hit, waiting before retrying...")
                time.sleep(20)
            else:
                raise e
    raise RuntimeError("Max retries exceeded for Alpha Vantage call.")

# ... rest of your unchanged code ...

if symbol:
    with st.spinner("Fetching data..."):
        try:
            df = safe_fetch(fetch_stock_data, symbol)
            df_weekly = safe_fetch(fetch_stock_data_weekly, symbol)
            df = analyze_data(df)
            fundamentals = fetch_fundamentals(symbol)
            options_data = fetch_options_data(symbol)
            news_items = fetch_news(symbol)
            sentiment_score = analyze_news_sentiment(news_items)
        except Exception as e:
            st.error(f"Error fetching or analyzing data: {e}")
            st.stop()



def fetch_options_data(symbol):
    return {
        "implied_volatility": 0.25,
        "put_call_ratio": 0.7,
        "open_interest_change": 0.05
    }

def fetch_fundamentals(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    return {
        "peRatio": info.get("trailingPE", None),
        "earningsDate": info.get("earningsDate", [None])[0] if isinstance(info.get("earningsDate"), list) else None,
        "revenueGrowth": info.get("revenueGrowth", None),
        "marketCap": info.get("marketCap", None),
    }

def fetch_news(symbol):
    try:
        query = f"{symbol} stock"
        articles = newsapi.get_everything(q=query, language="en", sort_by="publishedAt", page_size=5)
        return articles['articles']
    except Exception as e:
        st.error(f"NewsAPI error: {e}")
        return []

def analyze_news_sentiment(articles):
    polarity = 0
    for article in articles:
        if article.get("description"):
            blob = TextBlob(article["description"])
            polarity += blob.sentiment.polarity
    avg_polarity = polarity / len(articles) if articles else 0
    return avg_polarity

def analyze_data(df):
    df = dropna(df)
    df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    df["ATR"] = df["High"].combine(df["Low"], max) - df["Low"].combine(df["Close"], min)
    return df

def backtest_strategy(df):
    df = df.copy()
    df['Position'] = 0
    df.loc[df['momentum_rsi'] < 30, 'Position'] = 1
    df.loc[df['momentum_rsi'] > 70, 'Position'] = -1
    df['Market_Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Position'].shift(1) * df['Market_Returns']
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    df['Cumulative_Market_Returns'] = (1 + df['Market_Returns']).cumprod()
    return df

def plot_backtest(df):
    plt.figure(figsize=(10,5))
    plt.plot(df.index, df['Cumulative_Strategy_Returns'], label='Strategy Returns')
    plt.plot(df.index, df['Cumulative_Market_Returns'], label='Market Returns')
    plt.legend()
    st.pyplot(plt)

def risk_management(df):
    latest = df.iloc[-1]
    atr = latest.get("ATR")
    close = latest.get("Close")
    if atr and close:
        return close - 1.5 * atr, close + 3 * atr
    return None, None

def user_settings():
    st.sidebar.header("Customize Settings")
    rsi_buy = st.sidebar.slider("RSI Buy Threshold", 10, 50, 30)
    rsi_sell = st.sidebar.slider("RSI Sell Threshold", 50, 90, 70)
    sentiment_buy = st.sidebar.slider("Sentiment Buy Threshold", 0.0, 1.0, 0.3)
    sentiment_sell = st.sidebar.slider("Sentiment Sell Threshold", -1.0, 0.0, -0.3)
    return rsi_buy, rsi_sell, sentiment_buy, sentiment_sell

def prepare_ml_data(df):
    df = df.copy()
    df['Target'] = 0
    df.loc[df['Close'].shift(-1) > df['Close'], 'Target'] = 1
    df.loc[df['Close'].shift(-1) < df['Close'], 'Target'] = -1
    features = ['momentum_rsi', 'trend_macd', 'volume_adi', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl']
    df = df.dropna(subset=features + ['Target'])
    X = df[features]
    y = df['Target']
    return train_test_split(X, y, test_size=0.3, random_state=42)

def train_ml_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_ml_signal(model, X_test):
    return model.predict(X_test)

def generate_trade_signal(df, sentiment_score, rsi_buy, rsi_sell, sentiment_buy, sentiment_sell, ml_signal=None):
    latest = df.iloc[-1]
    rsi = latest.get("momentum_rsi")
    signal = "Hold"
    if rsi is not None:
        if rsi < rsi_buy:
            signal = "Buy"
        elif rsi > rsi_sell:
            signal = "Sell"
    if sentiment_score > sentiment_buy and signal == "Hold":
        signal = "Buy"
    elif sentiment_score < sentiment_sell and signal == "Hold":
        signal = "Sell"
    if ml_signal is not None:
        if ml_signal == 1:
            signal = "Buy"
        elif ml_signal == -1:
            signal = "Sell"
    return signal

st.title("📊 Enhanced Options Chat Assistant with Alpha Vantage (Free Endpoints)")

rsi_buy, rsi_sell, sentiment_buy, sentiment_sell = user_settings()
symbol = st.text_input("Enter a stock symbol (e.g., AAPL):", value="AAPL")

if symbol:
    with st.spinner("Fetching data..."):
        try:
            df = fetch_stock_data(symbol)
            df_weekly = fetch_stock_data_weekly(symbol)
            df = analyze_data(df)
            fundamentals = fetch_fundamentals(symbol)
            options_data = fetch_options_data(symbol)
            news_items = fetch_news(symbol)
            sentiment_score = analyze_news_sentiment(news_items)
        except Exception as e:
            st.error(f"Error fetching or analyzing data: {e}")
            st.stop()

    st.subheader("Fundamentals")
    st.write(fundamentals)
    st.subheader("Options Market Data (simulated)")
    st.write(options_data)

    st.subheader("Technical Analysis Summary (Daily)")
    st.write(df.tail(1).T)

    df_weekly = analyze_data(df_weekly)
    st.write("Weekly RSI:", df_weekly['momentum_rsi'].iloc[-1])

    st.subheader("Latest News")
    for article in news_items:
        st.markdown(f"### [{article['title']}]({article['url']})")
        st.write(article['description'])

    st.write(f"🧠 News Sentiment Score: `{sentiment_score:.2f}`")

    backtested_df = backtest_strategy(df)
    st.subheader("Backtesting Performance")
    plot_backtest(backtested_df)

    stop_loss, take_profit = risk_management(df)
    if stop_loss and take_profit:
        st.subheader("Risk Management Suggestions")
        st.write(f"Stop Loss: {stop_loss:.2f}")
        st.write(f"Take Profit: {take_profit:.2f}")

    X_train, X_test, y_train, y_test = prepare_ml_data(df)
    model = train_ml_model(X_train, y_train)
    ml_preds = predict_ml_signal(model, X_test)
    accuracy = accuracy_score(y_test, ml_preds)
    ml_signal = ml_preds[-1] if len(ml_preds) > 0 else None
    st.write(f"ML Model Accuracy on Test Set: {accuracy:.2f}")

    signal = generate_trade_signal(df, sentiment_score, rsi_buy, rsi_sell, sentiment_buy, sentiment_sell, ml_signal)
    st.success(f"📈 Trade Signal: **{signal}**")
