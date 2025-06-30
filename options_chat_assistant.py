import streamlit as st
import yfinance as yf
import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna
from newsapi import NewsApiClient
import subprocess
import sys
import datetime

# 1. Ensure textblob and nltk data are installed
try:
    from textblob import TextBlob
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "textblob"])
    from textblob import TextBlob
    import nltk
    nltk.download('brown')
    nltk.download('punkt')

# Additional libraries for ML & visualization
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

# Initialize NewsAPI client (replace with your own key)
newsapi = NewsApiClient(api_key='your_newsapi_key_here')

# === 1. Fetch stock data + extended timeframe for multi-timeframe analysis
def fetch_stock_data(symbol, period="6mo", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval)
    required_cols = {"Open", "High", "Low", "Close", "Volume"}

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.dropna(subset=required_cols)
    return df

def fetch_stock_data_weekly(symbol):
    # For multi timeframe analysis
    df = yf.download(symbol, period="1y", interval="1wk")
    return df

# === 2. Fetch options data (implied volatility, open interest, put-call ratio)
# This requires an external API; Yahoo does not provide it freely.
# For demo, we simulate with dummy data or skip if no API available.
def fetch_options_data(symbol):
    # Placeholder: In practice, fetch from Tradier, Alpha Vantage, or other
    # Return dict with IV skew, open interest etc.
    return {
        "implied_volatility": 0.25,
        "put_call_ratio": 0.7,
        "open_interest_change": 0.05
    }

# === 3. Fetch fundamental data (P/E ratio, earnings date, revenue growth)
def fetch_fundamentals(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    fundamentals = {
        "peRatio": info.get("trailingPE", None),
        "earningsDate": info.get("earningsDate", [None])[0],
        "revenueGrowth": info.get("revenueGrowth", None),
        "marketCap": info.get("marketCap", None),
    }
    return fundamentals

# === 4. Analyze news sentiment using TextBlob (you can replace with advanced NLP)
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

# === 5. Technical Analysis + Multiple Timeframes + Risk Metrics (ATR)
def analyze_data(df):
    try:
        df = dropna(df)
        df = add_all_ta_features(
            df,
            open="Open", high="High", low="Low", close="Close", volume="Volume",
            fillna=True
        )
        # Add ATR for risk management
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        df["ATR"] = high.combine(low, max) - low.combine(close, min)  # Simple ATR approx
        return df
    except Exception as e:
        st.error(f"TA-lib error: {e}")
        st.stop()

# === 6. Backtesting function for your strategy
def backtest_strategy(df):
    # Simple RSI based backtest as example
    df = df.copy()
    df['Position'] = 0
    df.loc[df['momentum_rsi'] < 30, 'Position'] = 1  # Buy
    df.loc[df['momentum_rsi'] > 70, 'Position'] = -1  # Sell

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

# === 7. Risk management suggestion
def risk_management(df):
    latest = df.iloc[-1]
    atr = latest.get("ATR", None)
    close = latest.get("Close", None)
    if atr and close:
        stop_loss = close - 1.5 * atr
        take_profit = close + 3 * atr
        return stop_loss, take_profit
    else:
        return None, None

# === 8. User customization - RSI and Sentiment thresholds
def user_settings():
    st.sidebar.header("Customize Settings")
    rsi_buy = st.sidebar.slider("RSI Buy Threshold", 10, 50, 30)
    rsi_sell = st.sidebar.slider("RSI Sell Threshold", 50, 90, 70)
    sentiment_buy = st.sidebar.slider("Sentiment Buy Threshold", 0.0, 1.0, 0.3)
    sentiment_sell = st.sidebar.slider("Sentiment Sell Threshold", -1.0, 0.0, -0.3)
    return rsi_buy, rsi_sell, sentiment_buy, sentiment_sell

# === 9. Alerts - Email placeholder (requires SMTP setup, here just a stub)
def send_alert(symbol, signal):
    st.info(f"Alert: {symbol} signal is {signal} (Add email/SMS integration here)")

# === 10. Machine Learning - Simple RandomForest classifier for buy/sell/hold
def prepare_ml_data(df):
    df = df.copy()
    df['Target'] = 0
    df.loc[df['Close'].shift(-1) > df['Close'], 'Target'] = 1  # 1 = price up next day
    df.loc[df['Close'].shift(-1) < df['Close'], 'Target'] = -1  # -1 = price down next day

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
    preds = model.predict(X_test)
    return preds

# Generate trade recommendation with user thresholds & ML signal
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

    # ML overrides if available
    if ml_signal is not None:
        if ml_signal == 1:
            signal = "Buy"
        elif ml_signal == -1:
            signal = "Sell"

    return signal

# Streamlit UI
st.title("📊 Enhanced Options Chat Assistant with Advanced Features")

# User Settings
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

    # Show fundamentals and options data
    st.subheader("Fundamentals")
    st.write(fundamentals)
    st.subheader("Options Market Data (simulated)")
    st.write(options_data)

    st.subheader("Technical Analysis Summary (Daily)")
    st.write(df.tail(1).T)

    # Multi timeframe RSI (weekly)
    df_weekly = analyze_data(df_weekly)
    st.write("Weekly RSI:", df_weekly['momentum_rsi'].iloc[-1])

    st.subheader("Latest News")
    for article in news_items:
        st.markdown(f"### [{article['title']}]({article['url']})")
        st.write(article['description'])

    st.write(f"🧠 News Sentiment Score: `{sentiment_score:.2f}`")

    # Backtesting strategy
    backtested_df = backtest_strategy(df)
    st.subheader("Backtesting Performance")
    plot_backtest(backtested_df)

    # Risk management levels
    stop_loss, take_profit = risk_management(df)
    if stop_loss and take_profit:
        st.subheader("Risk Management Suggestions")
        st.write(f"Stop Loss: {stop_loss:.2f}")
        st.write(f"Take Profit: {take_profit:.2f}")

    # ML model train and predict
    X_train, X_test, y_train, y_test = prepare_ml_data(df)
    model = train_ml_model(X_train, y_train)
    ml_preds = predict_ml_signal(model, X_test)
    accuracy = accuracy_score(y_test, ml_preds)
    ml_signal = ml_preds[-1] if len(ml_preds) > 0 else None
    st.write(f"ML Model Accuracy on Test Set: {accuracy:.2f}")

    # Generate trade recommendation with all signals
    signal = generate_trade_signal(df, sentiment_score, r
