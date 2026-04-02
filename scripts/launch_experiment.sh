#!/bin/bash
# Launch all 5 experiment arms as background processes
# Usage: ./scripts/launch_experiment.sh [train|trade|both]

cd /Users/kartikgarg/FinanceBot

MODE=${1:-trade}

if [ "$MODE" = "train" ] || [ "$MODE" = "both" ]; then
    echo "Training models for all timeframes..."
    python scripts/train_experiment.py --timeframes 5m 15m 1h 1d
    echo "Training complete."
fi

if [ "$MODE" = "trade" ] || [ "$MODE" = "both" ]; then
    echo ""
    echo "Launching experiment arms..."
    for TF in 5m 15m 1h 1d multi; do
        if pgrep -f "paper_trade_experiment.py --timeframe $TF" > /dev/null 2>&1; then
            echo "  $TF: already running"
        else
            mkdir -p data/experiments/$TF/paper_trading
            nohup python -u scripts/paper_trade_experiment.py --timeframe $TF \
                >> data/experiments/$TF/paper_trading/live_session.log 2>&1 &
            echo "  $TF: started (PID $!)"
            sleep 2  # Stagger launches to avoid yfinance rate limiting
        fi
    done
    echo ""
    echo "All arms launched. Monitor with:"
    echo "  tail -f data/experiments/*/paper_trading/live_session.log"
    echo "  python scripts/experiment_report.py"
fi
