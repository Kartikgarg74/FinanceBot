FROM python:3.11-slim

WORKDIR /app

# System deps for TA-Lib and other native packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make wget \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directories
RUN mkdir -p data/logs data/charts data/backtests

CMD ["python", "main.py", "--market", "all"]
