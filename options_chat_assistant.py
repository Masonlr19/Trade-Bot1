import os
import json
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
import requests
from openai import OpenAI

# -------------------- Config --------------------
st.set_page_config(page_title="Options Trading Chat Assistant", layout="wide")
st.title("📈 Options Trading Chat Assistant")

# Load OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")

# Watchlist JSON file path
WATCHLIST_FILE = "watchlist.json"

def load_watchlist():
    if not os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "w") as f:
            json.dump([], f)
    with open(WATCHLIST_FILE, "r") as f:
        return json.load(f)

def save_watchlist(watchlist):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(watchlist, f)

# -------------------- State Init --------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist()

# -------------------- Sidebar --------------------
st.sidebar.header("⭐ Watchlist")
new_ticker = st.sidebar.text_input("Add Ticker to Watchlist")
if st.sidebar.button("➕ Add"):
    if new_ticker and new_ticker.upper() not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_ticker.upper())
        save_watchlist(st.session_state.watchlist)
        st.sidebar.success(f"Added {new_ticker.upper()} to watchlist!")

if st.session_state.watchlist:
    selected_watch = st.sidebar.selectbox("📋 Your Tickers", st.session_state.watchlist)
    if st.sidebar.button("🔄 Load Ticker"):
        st.session_state.chat.append(("user", selected_watch))
    if st.sidebar.button("🗑️ Remove"):
        st.session_state.watchlist.remove(selected_watch)
        save_watchlist(st.session_state.watchlist)
        st.sidebar.warning(f"Removed {selected_watch}")

# -------------------- Functions --------------------
def fetch_option_chain(ticker):
    stock = yf.Ticker(ticker)
    try:
        exp_dates = stock.options
        options = stock.option_chain(exp_dates[0])
        calls = options.calls.head()
        return calls
    except Exception as e:
        return str(e)

def fetch_price_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo")
    df = add_technical_indicators(df)
    return df

def add_technical_indicators(df):
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    return df

def explain_with_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You're a helpful trading assistant who explains stock and option data in simple terms."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def explain_indicators(df, ticker):
    recent = df.iloc[-1]
    summary = f"""
Stock: {ticker}
- RSI: {recent['RSI']:.2f}
- MACD: {recent['MACD']:.2f}
- MACD Signal: {recent['MACD_signal']:.2f}
- SMA 20: {recent['SMA_20']:.2f}
- SMA 50: {recent['SMA_50']:.2f}
- Current Price: {recent['Close']:.2f}
"""

    prompt = f"""
You are a trading assistant. Interpret these technical indicators for {ticker} in plain English.
What do they say about momentum, overbought/oversold, and trend?

{summary}
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You're a helpful technical trading assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def get_alpha_vantage_news(ticker="AAPL"):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "apikey": ALPHA_VANTAGE_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    articles = data.get("feed", [])[:5]
    headlines = [f"{a['title']} - {a['summary']}" for a in articles]
    return headlines

def analyze_news_with_gpt(news_list, ticker):
    news_summary = "\n".join(news_list)
    prompt = f"""
Based on the following recent news headlines and summaries for {ticker}, suggest 2–3 option trades.
Include strategy (e.g. buy call/put, spread), expiration, and rationale.

{news_summary}
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You're a professional trading assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def plot_chart(df, strikes=None):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1,
        subplot_titles=("Stock Price with Strikes", "RSI")
    )

    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color='blue')), row=1, col=1)

    if strikes:
        for strike in strikes:
            fig.add_trace(go.Scatter(
                x=[df.index[0], df.index[-1]],
                y=[strike, strike],
                mode="lines",
                line=dict(dash='dot', color='gray'),
                name=f"Strike {strike}"
            ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)

    fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[70, 70], mode='lines', name='Overbought (70)',
                             line=dict(dash='dash', color='red')), row=2, col=1)

    fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[30, 30], mode='lines', name='Oversold (30)',
                             line=dict(dash='dash', color='green')), row=2, col=1)

    fig.update_layout(height=600, title_text="Price + RSI Chart", showlegend=True)
    return fig

# -------------------- Main Chat Logic --------------------
user_input = st.text_input("Ask a question or enter a stock symbol (e.g. AAPL):")

if user_input:
    st.session_state.chat.append(("user", user_input))
    ticker = user_input.upper() if len(user_input) <= 5 and user_input.isalpha() else None

    if ticker:
        data = fetch_option_chain(ticker)
        price_data = fetch_price_data(ticker)
        option_strikes = list(data['strike']) if not isinstance(data, str) else []

        if isinstance(data, str):
            response = f"⚠️ Error: {data}"
        else:
            explanation = explain_with_gpt(f"Explain this options data in simple terms:\n{data.to_string()}")
            indicators_explained = explain_indicators(price_data, ticker)
            response = f"Here are the first few call options for {ticker}:\n\n{data.to_string()}\n\n🧠 Explanation:\n{explanation}\n\n🧪 Technical Indicator Summary:\n{indicators_explained}"

            st.plotly_chart(plot_chart(price_data, option_strikes), use_container_width=True)

        if ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(ticker)
            save_watchlist(st.session_state.watchlist)

    else:
        response = explain_with_gpt(user_input)

    st.session_state.chat.append(("assistant", response))

# -------------------- Chat Display --------------------
for role, msg in st.session_state.chat:
    st.markdown(f"**{role.capitalize()}:** {msg}")

# -------------------- News-Based Trade Suggestions --------------------
st.header("🧠 News-Based Trade Suggestions (via Alpha Vantage)")

ticker_input = st.text_input("Enter a ticker for news analysis (e.g., TSLA)", value="AAPL")

if st.button("Get Trade Ideas"):
    headlines = get_alpha_vantage_news(ticker_input.upper())
    if not headlines:
        st.error("No news data found or API limit reached.")
    else:
        st.subheader("📚 Recent News:")
        for h in headlines:
            st.markdown(f"- {h}")
        st.subheader("📈 Suggested Trades:")
        trade_ideas = analyze_news_with_gpt(headlines, ticker_input.upper())
        st.write(trade_ideas)
