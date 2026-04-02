"""Central configuration for the multi-timeframe parallel experiment."""

from pathlib import Path

EXPERIMENT_START = "2026-04-01"
EXPERIMENT_END = "2026-06-01"
EXPERIMENT_DIR = Path("data/experiments")

# Top 20 performers from Phase 2 training (across all sectors)
EXPERIMENT_TICKERS = [
    "APOLLOHOSP", "INDUSINDBK", "NTPC", "TRENT", "DIVISLAB",
    "ONGC", "HINDALCO", "ADANIPORTS", "TCS", "HCLTECH",
    "RELIANCE", "MARUTI", "ITC", "CIPLA", "INFY",
    "WIPRO", "TATASTEEL", "ICICIBANK", "SBILIFE", "HDFCBANK",
]

TIMEFRAME_CONFIG = {
    "5m": {
        "interval": "5m",
        "days": 59,
        "sleep_sec": 300,
        "capital": 100000,
        "cost_mult": 0.5,      # 5m moves are tiny — lower threshold
        "min_samples": 100,
        "optuna_trials": 10,
        "max_tickers": 20,      # All 20 — 5m has enough time per cycle
    },
    "15m": {
        "interval": "15m",
        "days": 59,
        "sleep_sec": 900,
        "capital": 100000,
        "cost_mult": 0.75,
        "min_samples": 100,
        "optuna_trials": 10,
        "max_tickers": 20,
    },
    "1h": {
        "interval": "1h",
        "days": 59,
        "sleep_sec": 3600,
        "capital": 100000,
        "cost_mult": 1.0,
        "min_samples": 100,
        "optuna_trials": 10,
        "max_tickers": 20,
    },
    "1d": {
        "interval": "1d",
        "days": 365,
        "sleep_sec": 86400,     # Once per day
        "capital": 100000,
        "cost_mult": 2.0,
        "min_samples": 80,      # Daily has fewer bars
        "optuna_trials": 5,     # Less data → fewer trials
        "max_tickers": 20,
    },
    "multi": {
        "interval": "15m",      # Primary entry timeframe
        "days": 59,
        "sleep_sec": 900,       # Runs on 15m cycle
        "capital": 100000,
        "cost_mult": 0.75,
        "min_samples": 100,
        "optuna_trials": 10,
        "max_tickers": 20,
    },
}


def get_arm_dirs(arm: str) -> dict:
    """Get all paths for an experiment arm."""
    base = EXPERIMENT_DIR / arm
    return {
        "base": base,
        "models": base / "models",
        "paper_trading": base / "paper_trading",
        "reports": base / "reports",
        "log": base / "paper_trading" / "live_session.log",
        "feedback": base / "paper_trading" / "feedback_buffer.json",
        "trades_csv": base / "paper_trading" / "paper_trades.csv",
        "snapshots": base / "paper_trading" / "paper_snapshots.json",
    }
