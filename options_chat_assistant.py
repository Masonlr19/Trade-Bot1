import streamlit as st
import pandas as pd
import yfinance as yf
from ta import add_all_ta_features
from ta.utils import dropna
from newsapi import NewsApiClient
import subprocess
import sys
import datetime
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import yfinance
st.write("yfinance version:", yfinance.__version__)

# Initialize NewsAPI client
newsapi = NewsApiClient(api_key='your_newsapi_key_here')

def fetch_stock_data(symbol, period="1y", interval="1d"):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        raise ValueError("No data fetched from Yahoo Finance.")
    df.rename(columns={
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume",
        "Dividends": "Dividend"
    }, inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

def fetch_stock_data_weekly(symbol, period="2y"):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval="1wk")
    if df.empty:
        raise ValueError("No weekly data fetched from Yahoo Finance.")
    df.rename(columns={
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume",
        "Dividends": "Dividend"
    }, inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

def fetch_options_data(symbol):
    ticker = yf.Ticker(symbol)
    try:
        expirations = ticker.options
        if not expirations:
            return {"implied_volatility": None, "put_call_ratio": None, "open_interest_change": None}
        
        # Pick nearest expiration date
        expiry = expirations[0]
        opt_chain = ticker.option_chain(expiry)
        
        calls = opt_chain.calls
        puts = opt_chain.puts

        # Calculate some aggregates (mean implied vol, put/call volume ratio, open interest change simulated)
        mean_iv_calls = calls['impliedVolatility'].mean() if not calls.empty else None
        mean_iv_puts = puts['impliedVolatility'].mean() if not puts.empty else None
        
        # Put/Call volume ratio
        put_vol = puts['volume'].sum() if not puts.empty else 0
        call_vol = calls['volume'].sum() if not calls.empty else 0
        put_call_ratio = put_vol / call_vol if call_vol > 0 else None

        # Open interest change is not available directly, so simulate by checking last vs prior day
        # We'll skip for simplicity or just use current open interest averages
        avg_oi_calls = calls['openInterest'].mean() if not calls.empty else None
        avg_oi_puts = puts['openInterest'].mean() if not puts.empty else None
        open_interest_change = None  # Placeholder, no real data

        return {
            "mean_iv_calls": mean_iv_calls,
            "mean_iv_puts": mean_iv_puts,
            "put_call_ratio": put_call_ratio,
            "avg_oi_calls": avg_oi_calls,
            "avg_oi_puts": avg_oi_puts,
            "open_interest_change": open_interest_change
        }
    except Exception as e:
        st.warning(f"Error fetching options data: {e}")
        return {"implied_volatility": None, "put_call_ratio": None, "open_interest_change": None}

def fetch_fundamentals(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    return {
        "peRatio": info.get("trailingPE", None),
        "earningsDate": info.get("earningsDate", [None])[0] if isinstance(info.get("earningsDate"), list) else info.get("earningsDate"),
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
    if 'momentum_rsi' not in df.columns or 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'momentum_rsi' and 'Close' columns")

    df = df.dropna(subset=['momentum_rsi', 'Close'])

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
    required_cols = ['Close', 'momentum_rsi', 'trend_macd', 'volume_adi', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Create target variable
    close = df['Close'].values
    target = []
    for i in range(len(close) - 1):
        if close[i + 1] > close[i]:
            target.append(1)
        elif close[i + 1] < close[i]:
            target.append(-1)
        else:
            target.append(0)
    target.append(0)  # for last row

    # Now assign target to df safely
    df = df.iloc[:len(target)].copy()
    df['Target'] = target

    X = df[required_cols[1:]]  # exclude 'Close'
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

# Streamlit UI
st.title("📊 Enhanced Options Chat Assistant with yfinance")

rsi_buy, rsi_sell, sentiment_buy, sentiment_sell = user_settings()

symbol = st.text_input("Enter stock symbol (e.g., AAPL):").upper()

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

            # ML
            X_train, X_test, y_train, y_test = prepare_ml_data(df)
            model = train_ml_model(X_train, y_train)
            ml_predictions = predict_ml_signal(model, X_test)
            accuracy = accuracy_score(y_test, ml_predictions)
            ml_signal = ml_predictions[-1] if len(ml_predictions) > 0 else None

            trade_signal = generate_trade_signal(df, sentiment_score, rsi_buy, rsi_sell, sentiment_buy, sentiment_sell, ml_signal)

            # Show results
            st.subheader(f"Trade Signal: {trade_signal}")
            st.write(f"RSI Buy: {rsi_buy}, RSI Sell: {rsi_sell}")
            st.write(f"Sentiment Score: {sentiment_score:.3f}")
            st.write(f"ML Model Accuracy: {accuracy:.2%}")
            st.write("Fundamentals:")
            st.json(fundamentals)
            st.write("Options Data:")
            st.json(options_data)

            st.write("News Headlines:")
            for article in news_items:
                st.markdown(f"**{article['title']}**")
                st.write(article['description'])
                st.write(f"[Read more]({article['url']})")
                st.write("---")

            backtest_df = backtest_strategy(df)
            plot_backtest(backtest_df)

        except Exception as e:
            st.error(f"Error fetching or analyzing data: {e}")
