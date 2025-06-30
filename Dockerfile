# Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for torch & transformers (some base libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY financial_advisor_bot.py ./

# Expose Streamlit port and set environment variable example (override at runtime!)
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Command to run the app
CMD ["streamlit", "run", "financial_advisor_bot.py", "--server.port=8501", "--server.address=0.0.0.0"]
