#!/bin/bash
# Gracefully stop all experiment arms
echo "Stopping experiment arms..."
pkill -f "paper_trade_experiment.py" 2>/dev/null
echo "All arms stopped."
echo "Final logs at: data/experiments/*/paper_trading/live_session.log"
