#!/usr/bin/env python3
"""
Paper Trading Loop — Live simulation with ML signals.

Detects optimal candle interval from data, runs ML inference on schedule,
executes paper trades with realistic fees, and logs everything.

The scheduler:
1. Fetches latest OHLCV data
2. Runs feature pipeline
3. Gets ML prediction (ensemble or tree model)
4. Executes paper trade via PaperTradingEngine (with Zerodha fees)
5. Logs trade + sends alert
6. Repeats at the detected candle interval

Usage:
    python scripts/paper_trade.py --tickers RELIANCE TCS INFY --interval 1h
    python scripts/paper_trade.py --tickers RELIANCE --interval auto
    python scripts/paper_trade.py --config paper_config.json
"""

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.finance.analysis.data_fetcher import DataFetcher
from src.finance.paper_engine import PaperTradingEngine
from src.finance.cost_model import estimate_round_trip_pct
from src.finance.market_hours import MarketHours
from src.ml.feature_pipeline import FeaturePipeline
from src.ml.models import TradingModelTrainer
from src.ml.online_learner import OnlineLearner, TradeFeedback
from src.finance.base_trader import Signal, SignalAction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/paper_trading.log"),
    ],
)
logger = logging.getLogger("paper_trade")

MODEL_DIR = Path("data/ml_models")
PHASE2_DIR = Path("data/ml_models/phase2")
DATA_DIR = Path("data/paper_trading")

# Graceful shutdown
running = True
def signal_handler(sig, frame):
    global running
    logger.info("Shutdown signal received. Finishing current cycle...")
    running = False
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def detect_candle_interval(df: pd.DataFrame) -> int:
    """
    Detect the average candle formation time in minutes from the data.

    Returns the interval in minutes (e.g., 60 for 1h candles, 15 for 15m).
    """
    if df is None or len(df) < 2:
        return 60  # Default 1 hour

    # Calculate time differences between consecutive bars
    idx = df.index
    if hasattr(idx, 'to_series'):
        times = idx.to_series()
    else:
        times = pd.Series(idx)

    diffs = times.diff().dropna()

    if len(diffs) == 0:
        return 60

    # Get median difference in minutes (median is more robust than mean)
    median_diff = diffs.median()

    if hasattr(median_diff, 'total_seconds'):
        minutes = int(median_diff.total_seconds() / 60)
    else:
        # If it's a Timedelta
        minutes = int(pd.Timedelta(median_diff).total_seconds() / 60)

    # Snap to standard intervals
    standard = [1, 5, 15, 30, 60, 240, 1440]  # 1m, 5m, 15m, 30m, 1h, 4h, 1d
    closest = min(standard, key=lambda x: abs(x - minutes))

    logger.info("Detected candle interval: %d minutes (raw: %d min)", closest, minutes)
    return closest


def load_model(ticker: str):
    """Load the best available model for a ticker (Phase 2 ensemble > Phase 1 tree)."""
    safe = ticker.replace("/", "_")

    # Try Phase 2 tree model first (ensemble causes segfault on macOS)
    phase2_tree = PHASE2_DIR / f"{safe}_lightgbm.joblib"
    if phase2_tree.exists():
        try:
            trainer = TradingModelTrainer()
            model = trainer.load_model(phase2_tree)
            meta_path = PHASE2_DIR / f"{safe}_metadata.json"
            metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            logger.info("Loaded Phase 2 tree for %s", ticker)
            return model, metadata, "tree"
        except Exception as e:
            logger.warning("Failed to load Phase 2 tree for %s: %s", ticker, e)

    # Fall back to Phase 1 tree model
    for model_file in sorted(MODEL_DIR.glob(f"{safe}_*.joblib"), reverse=True):
        try:
            trainer = TradingModelTrainer()
            model = trainer.load_model(model_file)
            meta_path = MODEL_DIR / f"{safe}_metadata.json"
            metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            logger.info("Loaded Phase 1 model for %s: %s", ticker, model_file.name)
            return model, metadata, "tree"
        except Exception as e:
            logger.warning("Failed to load %s: %s", model_file, e)

    logger.error("No model found for %s", ticker)
    return None, {}, None


def generate_ml_signal(model, metadata: dict, model_type: str,
                       df: pd.DataFrame, ticker: str) -> Signal:
    """Generate a trading signal from the loaded model."""
    pipeline = FeaturePipeline()

    try:
        features = pipeline.transform(df)
        features = features.replace([np.inf, -np.inf], np.nan).dropna()

        if len(features) == 0:
            return Signal(ticker=ticker, action=SignalAction.HOLD, confidence=0, price=0,
                         reasoning="No valid features")

        # Filter to trained features
        trained_feats = metadata.get("features", [])
        if trained_feats:
            available = [f for f in trained_feats if f in features.columns]
            if len(available) < len(trained_feats) * 0.8:
                return Signal(ticker=ticker, action=SignalAction.HOLD, confidence=0, price=0,
                             reasoning=f"Missing {len(trained_feats)-len(available)} features")
            features = features[available]

        price = float(df["Close"].iloc[-1])

        if model_type == "ensemble":
            preds, probs = model.predict(features.iloc[[-1]])
            prediction = int(preds[0])
            confidence = float(probs[0].max()) * 100 if probs is not None else 50
        else:
            # Tree model
            pred_raw = model.predict(features.iloc[[-1]])
            label_map = {0: -1, 1: 0, 2: 1}
            prediction = label_map[int(pred_raw[0])]
            confidence = 50
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features.iloc[[-1]])
                confidence = float(probs[0].max()) * 100

        # Determine action — use ARGMAX (best class wins), not confidence threshold
        # The model's job is to pick the best action; our job is to act on it
        # Only HOLD if the model's #1 prediction IS hold, or if BUY/SELL prob < 25%
        if prediction == 1 and confidence >= 25:
            action = SignalAction.BUY
        elif prediction == -1 and confidence >= 25:
            action = SignalAction.SELL
        else:
            # Even if model says HOLD, check if BUY or SELL prob > HOLD prob
            if model_type == "tree" and hasattr(model, "predict_proba"):
                sell_p, hold_p, buy_p = probs[0][0], probs[0][1], probs[0][2]
                if buy_p > hold_p and buy_p > sell_p and buy_p >= 0.30:
                    action = SignalAction.BUY
                    prediction = 1
                    confidence = buy_p * 100
                elif sell_p > hold_p and sell_p > buy_p and sell_p >= 0.30:
                    action = SignalAction.SELL
                    prediction = -1
                    confidence = sell_p * 100
                else:
                    action = SignalAction.HOLD
            else:
                action = SignalAction.HOLD

        # ATR for stop-loss
        import pandas_ta as ta
        atr = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        atr_val = float(atr.iloc[-1]) if atr is not None and not pd.isna(atr.iloc[-1]) else price * 0.02

        if action == SignalAction.BUY:
            sl = round(price - 2 * atr_val, 2)
            tp = round(price + 3 * atr_val, 2)
        elif action == SignalAction.SELL:
            sl = round(price + 2 * atr_val, 2)
            tp = round(price - 3 * atr_val, 2)
        else:
            sl = tp = None

        return Signal(
            ticker=ticker, action=action, confidence=round(confidence, 1),
            price=price, stop_loss=sl, take_profit=tp,
            reasoning=f"ML:{action.value}(conf={confidence:.0f}%,model={model_type})",
        )

    except Exception as e:
        logger.error("Signal generation failed for %s: %s", ticker, e)
        return Signal(ticker=ticker, action=SignalAction.HOLD, confidence=0, price=0,
                     reasoning=f"Error: {e}")


def run_paper_trading_loop(
    tickers: list[str],
    interval: str = "1h",
    initial_capital: float = 100000,
    broker: str = "zerodha",
    trade_type: str = "intraday",
):
    """Main paper trading loop."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fetcher = DataFetcher()

    # Load models for all tickers
    models = {}
    for ticker in tickers:
        model, meta, mtype = load_model(ticker)
        if model:
            models[ticker] = (model, meta, mtype)
        else:
            logger.warning("Skipping %s — no model available", ticker)

    if not models:
        logger.error("No models loaded. Exiting.")
        return

    # Initialize paper trading engine
    engine = PaperTradingEngine(
        broker=broker,
        trade_type=trade_type,
        initial_capital=initial_capital,
        slippage_bps=10,
        max_position_pct=15.0,
        max_positions=len(tickers),
    )

    # Initialize self-learning system
    feedback = TradeFeedback(DATA_DIR / "feedback_buffer.json")
    learner = OnlineLearner(
        model_dir=PHASE2_DIR,
        feedback_buffer=feedback,
        retrain_threshold=20,  # Retrain after every 20 closed trades
    )

    # Detect candle interval
    sample_ticker = list(models.keys())[0]
    sample_df = fetcher.fetch_indian_stock(sample_ticker, interval, 10)
    if interval == "auto" and sample_df is not None:
        interval_minutes = detect_candle_interval(sample_df)
    else:
        interval_map = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
        interval_minutes = interval_map.get(interval, 60)

    logger.info("=" * 60)
    logger.info("PAPER TRADING STARTED")
    logger.info("  Tickers: %s", ", ".join(models.keys()))
    logger.info("  Interval: %s (%d minutes)", interval, interval_minutes)
    logger.info("  Capital: %s %s", f"{initial_capital:,.0f}",
                "INR" if broker == "zerodha" else "USD")
    logger.info("  Broker: %s (%s)", broker, trade_type)
    logger.info("  Cost: %.3f%% round-trip", estimate_round_trip_pct(broker) * 100)
    logger.info("=" * 60)

    cycle = 0
    while running:
        cycle += 1
        cycle_start = datetime.now()
        logger.info("\n--- Cycle %d at %s ---", cycle, cycle_start.strftime("%Y-%m-%d %H:%M:%S"))

        signals_generated = 0
        trades_executed = 0

        for ticker in models:
            model, meta, mtype = models[ticker]

            # Fetch latest data
            try:
                df = fetcher.fetch_indian_stock(ticker, interval, 59)
                if df is None or len(df) < 60:
                    logger.warning("[%s] Insufficient data (%s bars)", ticker,
                                   len(df) if df is not None else 0)
                    continue
            except Exception as e:
                logger.error("[%s] Data fetch failed: %s", ticker, e)
                continue

            # Generate signal
            sig = generate_ml_signal(model, meta, mtype, df, ticker)
            signals_generated += 1

            current_price = float(df["Close"].iloc[-1])

            if sig.action != SignalAction.HOLD:
                logger.info("[%s] SIGNAL: %s (conf=%.1f%%, price=%.2f)",
                           ticker, sig.action.value, sig.confidence, current_price)

                # Execute paper trade
                trade = engine.execute_signal(sig, current_price)
                if trade:
                    trades_executed += 1
                    logger.info("[%s] TRADE EXECUTED: %s | Net P&L: %+.2f",
                               ticker, trade.side, trade.net_pnl)

                    # Feed outcome to self-learning system
                    predicted_action = 1 if sig.action == SignalAction.BUY else -1
                    learner.on_trade_closed(
                        ticker=ticker,
                        features_dict={},  # Features already logged
                        predicted_action=predicted_action,
                        net_pnl=trade.net_pnl,
                        confidence=sig.confidence,
                    )
            else:
                logger.debug("[%s] HOLD (conf=%.1f%%)", ticker, sig.confidence)

        # Take daily snapshot
        snapshot = engine.take_snapshot()
        logger.info("Portfolio: capital=%.2f, positions=%d, trades=%d, drawdown=%.1f%%",
                    snapshot["capital"], snapshot["positions"],
                    snapshot["total_trades"], snapshot["drawdown_pct"])

        # Save state periodically (every 5 cycles)
        if cycle % 5 == 0:
            engine.save_trades(DATA_DIR / "paper_trades.csv")
            with open(DATA_DIR / "paper_snapshots.json", "w") as f:
                json.dump(engine.daily_snapshots, f, indent=2)
            logger.info("State saved (%d trades, %d snapshots)",
                       len(engine.trade_history), len(engine.daily_snapshots))

            # Log learning progress
            learning_report = learner.get_learning_report()
            logger.info("Self-Learning Status:\n%s", learning_report)

        # Wait for next cycle
        elapsed = (datetime.now() - cycle_start).total_seconds()
        sleep_time = max(0, interval_minutes * 60 - elapsed)

        if not running:
            break

        if sleep_time > 0:
            logger.info("Next cycle in %.0f minutes (%.0fs)", sleep_time / 60, sleep_time)
            # Sleep in 10s chunks so we can respond to signals
            for _ in range(int(sleep_time / 10)):
                if not running:
                    break
                time.sleep(10)
            remaining = sleep_time % 10
            if remaining > 0 and running:
                time.sleep(remaining)

    # Final save
    logger.info("\n" + "=" * 60)
    logger.info("PAPER TRADING STOPPED")
    logger.info("=" * 60)

    engine.save_trades(DATA_DIR / "paper_trades.csv")
    with open(DATA_DIR / "paper_snapshots.json", "w") as f:
        json.dump(engine.daily_snapshots, f, indent=2)

    report = engine.get_performance_report()
    print("\n" + report)

    report_path = DATA_DIR / "paper_report.txt"
    report_path.write_text(report)
    logger.info("Final report saved: %s", report_path)


def main():
    parser = argparse.ArgumentParser(description="Paper Trading with ML Signals")
    from src.ml.ticker_config import INDIA_TICKERS_FLAT
    parser.add_argument("--tickers", nargs="+", default=INDIA_TICKERS_FLAT)
    parser.add_argument("--interval", default="1h", help="Candle interval (1h, 15m, etc.) or 'auto'")
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--broker", default="zerodha")
    parser.add_argument("--trade-type", default="intraday")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit (for testing)")
    args = parser.parse_args()

    if args.once:
        # Single cycle for testing
        global running
        run_paper_trading_loop(
            tickers=args.tickers,
            interval=args.interval,
            initial_capital=args.capital,
            broker=args.broker,
            trade_type=args.trade_type,
        )
        running = False
    else:
        run_paper_trading_loop(
            tickers=args.tickers,
            interval=args.interval,
            initial_capital=args.capital,
            broker=args.broker,
            trade_type=args.trade_type,
        )


if __name__ == "__main__":
    main()
