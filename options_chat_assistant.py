import yfinance as yf
import streamlit as st
from openai import OpenAI
import os

# Create the OpenAI client using the updated SDK
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
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You're a helpful trading assistant who explains stock and option data in simple terms."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error with OpenAI API: {e}"

# Chat logic
if user_input:
    st.session_state.chat.append(("user", user_input))

    if len(user_input) <= 5 and user_input.isalpha():
        data = fetch_option_chain(user_input.upper())
        if isinstance(data, str):
            response = f"⚠️ {data}"
        else:
            explanation = explain_with_gpt(f"Explain this options data in simple terms:\n{data.to_string()}")
            response = f"Here are the first few call options for {user_input.upper()}:\n\n{data.to_string()}\n\n🧠 Explanation:\n{explanation}"
    else:
        explanation = explain_with_gpt(user_input)
        response = explanation

    st.session_state.chat.append(("assistant", response))

# Display chat
for role, msg in st.session_state.chat:
    st.markdown(f"**{role.capitalize()}:** {msg}")
