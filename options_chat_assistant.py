import yfinance as yf
import streamlit as st
from openai import OpenAI
import os

# Create the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Options Chat Assistant", layout="wide")
st.title("📈 Options Trading Chat Assistant")

# Chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

# User input
user_input = st.text_input("Ask a question or enter a stock symbol (e.g. AAPL):")

def fetch_option_chain(ticker):
    stock = yf.Ticker(ticker)
    try:
        exp_dates = stock.options
        if not exp_dates:
            return "No options data available for this stock."
        options = stock.option_chain(exp_dates[0])
        calls = options.calls.head()
        return calls
    except Exception as e:
        return f"Error fetching options data: {e}"

def explain_with_gpt(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using GPT-3.5 here
            messages=[
                {"role": "system", "content": "You're a helpful trading assistant who explains stock and option data in simple terms."
