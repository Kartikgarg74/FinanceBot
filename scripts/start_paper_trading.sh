#!/bin/bash
# Start paper trading bot if not already running
# Add to crontab for daily auto-start:
#   0 8 * * 1-5 /Users/kartikgarg/FinanceBot/scripts/start_paper_trading.sh

cd /Users/kartikgarg/FinanceBot

# Check if already running
if pgrep -f "paper_trade.py" > /dev/null; then
    echo "$(date): Paper trader already running (PID $(pgrep -f paper_trade.py))"
    exit 0
fi

# Start paper trader
echo "$(date): Starting paper trader..."
nohup python -u scripts/paper_trade.py \
    --tickers APOLLOHOSP INDUSINDBK NTPC TRENT DIVISLAB ONGC HINDALCO ADANIPORTS TCS HCLTECH RELIANCE MARUTI ITC CIPLA INFY WIPRO TATASTEEL ICICIBANK SBILIFE HDFCBANK \
    --interval 1h \
    >> data/paper_trading/live_session.log 2>&1 &

echo "$(date): Started with PID $!"
echo "$(date): Started with PID $!" >> data/paper_trading/startup.log
