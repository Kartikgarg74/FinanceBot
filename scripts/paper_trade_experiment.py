#!/usr/bin/env python3
"""Timeframe-aware paper trader for the multi-timeframe experiment.

Usage:
    python scripts/paper_trade_experiment.py --timeframe 5m
    python scripts/paper_trade_experiment.py --timeframe multi
"""

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas_ta as ta
from src.finance.analysis.data_fetcher import DataFetcher
from src.finance.paper_engine import PaperTradingEngine
from src.finance.base_trader import Signal, SignalAction
from src.ml.feature_pipeline import FeaturePipeline
from src.ml.models import TradingModelTrainer
from src.ml.online_learner import OnlineLearner, TradeFeedback
from src.finance.market_hours import MarketHours
from scripts.experiment_config import TIMEFRAME_CONFIG, EXPERIMENT_TICKERS, get_arm_dirs

running = True
def signal_handler(sig, frame):
    global running
    running = False
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def load_models(model_dir: Path) -> dict:
    """Load all LightGBM models from a directory."""
    import joblib
    models = {}
    for jf in model_dir.glob("*_lightgbm.joblib"):
        ticker = jf.stem.replace("_lightgbm", "")
        meta_path = model_dir / f"{ticker}_metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        models[ticker] = (joblib.load(jf), meta)
    return models


def predict_single(model, meta: dict, df, pipeline: FeaturePipeline) -> tuple[SignalAction, float, float]:
    """Run prediction for a single ticker. Returns (action, confidence, price)."""
    features = pipeline.transform(df)
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    if len(features) == 0:
        return SignalAction.HOLD, 0, 0

    trained = meta.get("features", [])
    if trained:
        avail = [f for f in trained if f in features.columns]
        if len(avail) < len(trained) * 0.8:
            return SignalAction.HOLD, 0, 0
        features = features[avail]

    X = features.iloc[[-1]]
    price = float(df["Close"].iloc[-1])

    pred_raw = model.predict(X)[0]
    label_map = {0: -1, 1: 0, 2: 1}
    prediction = label_map[int(pred_raw)]
    confidence = 50.0

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        confidence = float(probs.max()) * 100
        sell_p, hold_p, buy_p = probs[0], probs[1], probs[2]
        if buy_p > hold_p and buy_p > sell_p and buy_p >= 0.30:
            prediction = 1
            confidence = buy_p * 100
        elif sell_p > hold_p and sell_p > buy_p and sell_p >= 0.30:
            prediction = -1
            confidence = sell_p * 100

    if prediction == 1 and confidence >= 25:
        return SignalAction.BUY, confidence, price
    elif prediction == -1 and confidence >= 25:
        return SignalAction.SELL, confidence, price
    return SignalAction.HOLD, confidence, price


def run_arm(timeframe: str):
    """Run a single experiment arm."""
    cfg = TIMEFRAME_CONFIG[timeframe]
    dirs = get_arm_dirs(timeframe)
    interval = cfg["interval"]
    sleep_sec = cfg["sleep_sec"]

    # Setup logging to arm-specific file
    log_path = dirs["log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger = logging.getLogger(f"arm_{timeframe}")
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    fetcher = DataFetcher()
    pipeline = FeaturePipeline()

    # Load models
    if timeframe == "multi":
        from src.ml.multi_timeframe_signal import MultiTimeframeSignalGenerator
        model_dirs = {
            "1d": get_arm_dirs("1d")["models"],
            "1h": get_arm_dirs("1h")["models"],
            "15m": get_arm_dirs("15m")["models"],
        }
        multi_gen = MultiTimeframeSignalGenerator(model_dirs, fetcher)
        models = {t: None for t in EXPERIMENT_TICKERS}  # Placeholder
        logger.info("Multi-timeframe generator loaded (1d + 1h + 15m)")
    else:
        models = load_models(dirs["models"])
        multi_gen = None
        logger.info("Loaded %d models for %s arm", len(models), timeframe)

    if not models:
        logger.error("No models found for %s arm", timeframe)
        return

    # Paper engine
    engine = PaperTradingEngine(
        broker="zerodha", trade_type="intraday",
        initial_capital=cfg["capital"], slippage_bps=10,
        max_position_pct=15.0, max_positions=len(models),
    )

    # Self-learning
    feedback = TradeFeedback(dirs["feedback"])
    learner = OnlineLearner(
        model_dir=dirs["models"], feedback_buffer=feedback,
        retrain_threshold=20, interval=interval, days=cfg["days"],
    )

    # Market hours check (NSE: 9:15 AM - 3:30 PM IST)
    market_hours = MarketHours("india", {"schedule": {"market_hours": "09:15-15:30", "timezone": "Asia/Kolkata"}})

    # Track last seen prices to detect stale data
    last_prices: dict[str, float] = {}

    logger.info("=" * 60)
    logger.info("EXPERIMENT ARM: %s | Interval: %s | Capital: Rs %s",
                timeframe, interval, f"{cfg['capital']:,}")
    logger.info("Tickers: %d | Sleep: %ds | Market hours enforced", len(models), sleep_sec)
    logger.info("=" * 60)

    cycle = 0
    while running:
        cycle += 1
        start = datetime.now()

        # FIX 1: Skip if market is closed (except 1d which runs after close)
        if timeframe != "1d" and not market_hours.is_open():
            logger.debug("Market closed — skipping cycle %d", cycle)
            # Still sleep but don't process
            for _ in range(int(sleep_sec / 10)):
                if not running:
                    break
                import time as _time
                _time.sleep(10)
            continue

        logger.info("--- Cycle %d at %s ---", cycle, start.strftime("%H:%M:%S"))

        trades_count = 0

        # FIX 2: Check stop-loss / take-profit on open positions FIRST
        for ticker, pos in list(engine.positions.items()):
            try:
                df_check = fetcher.fetch_indian_stock(ticker, interval, 5)
                if df_check is None or df_check.empty:
                    continue
                current_price = float(df_check["Close"].iloc[-1])

                # Check if stop-loss or take-profit hit
                should_close = False
                close_reason = ""

                # ATR-based stop-loss (2x ATR from entry)
                import pandas_ta as ta
                atr_check = ta.atr(df_check["High"], df_check["Low"], df_check["Close"], length=14)
                atr_val = float(atr_check.iloc[-1]) if atr_check is not None and len(atr_check) > 0 and not np.isnan(atr_check.iloc[-1]) else current_price * 0.02

                loss_pct = (current_price - pos.avg_price) / pos.avg_price
                if loss_pct < -0.03:  # 3% stop-loss
                    should_close = True
                    close_reason = f"STOP-LOSS hit ({loss_pct:.1%})"
                elif loss_pct > 0.045:  # 4.5% take-profit
                    should_close = True
                    close_reason = f"TAKE-PROFIT hit ({loss_pct:.1%})"

                # Time-based exit: close after 20 cycles of holding
                hold_hours = (datetime.now() - pos.entry_time).total_seconds() / 3600
                max_hold = {"5m": 2, "15m": 6, "1h": 24, "1d": 120, "multi": 6}.get(timeframe, 24)
                if hold_hours > max_hold:
                    should_close = True
                    close_reason = f"TIME-EXIT ({hold_hours:.1f}h > {max_hold}h max)"

                if should_close:
                    sell_sig = Signal(ticker=ticker, action=SignalAction.SELL,
                                    confidence=100, price=current_price,
                                    reasoning=close_reason)
                    trade = engine.execute_signal(sell_sig, current_price)
                    if trade:
                        trades_count += 1
                        logger.info("[%s] AUTO-EXIT: %s | PnL: %+.2f", ticker, close_reason, trade.net_pnl)
                        pred_act = 1  # Was a long position
                        learner.on_trade_closed(ticker, {}, pred_act, trade.net_pnl, 100)
            except Exception as e:
                logger.warning("[%s] SL/TP check error: %s", ticker, e)

        # Now process new signals
        for ticker in list(models.keys()):
            try:
                if timeframe == "multi":
                    sig = multi_gen.generate(ticker)
                else:
                    model, meta = models[ticker]
                    df = fetcher.fetch_indian_stock(ticker, interval, cfg["days"])
                    if df is None or len(df) < 60:
                        continue

                    current_price = float(df["Close"].iloc[-1])

                    # FIX 3: Detect stale data — skip if price unchanged from last cycle
                    prev_price = last_prices.get(ticker)
                    if prev_price is not None and abs(current_price - prev_price) < 0.01:
                        continue  # Stale data, skip
                    last_prices[ticker] = current_price

                    action, conf, price = predict_single(model, meta, df, pipeline)
                    atr = ta.atr(df["High"], df["Low"], df["Close"], length=14)  # noqa: already imported
                    atr_v = float(atr.iloc[-1]) if atr is not None and not np.isnan(atr.iloc[-1]) else price * 0.02
                    sl = tp = None
                    if action == SignalAction.BUY:
                        sl, tp = round(price - 2*atr_v, 2), round(price + 3*atr_v, 2)
                    elif action == SignalAction.SELL:
                        sl, tp = round(price + 2*atr_v, 2), round(price - 3*atr_v, 2)
                    sig = Signal(ticker=ticker, action=action, confidence=round(conf, 1),
                                price=price, stop_loss=sl, take_profit=tp,
                                reasoning=f"{timeframe}:{'BUY' if action==SignalAction.BUY else 'SELL' if action==SignalAction.SELL else 'HOLD'}")

                if sig.action != SignalAction.HOLD:
                    logger.info("[%s] SIGNAL: %s (conf=%.1f%%, price=%.2f)",
                               ticker, sig.action.value, sig.confidence, sig.price)
                    trade = engine.execute_signal(sig, sig.price)
                    if trade:
                        trades_count += 1
                        logger.info("[%s] TRADE: %s | PnL: %+.2f", ticker, trade.side, trade.net_pnl)
                        pred_act = 1 if sig.action == SignalAction.BUY else -1
                        learner.on_trade_closed(ticker, {}, pred_act, trade.net_pnl, sig.confidence)
            except Exception as e:
                logger.warning("[%s] Error: %s", ticker, e)

        snap = engine.take_snapshot()
        logger.info("Portfolio: capital=%.2f pos=%d trades=%d dd=%.1f%% | Cycle trades: %d",
                    snap["capital"], snap["positions"], snap["total_trades"],
                    snap["drawdown_pct"], trades_count)

        # Save periodically
        if cycle % 5 == 0:
            engine.save_trades(dirs["trades_csv"])
            with open(dirs["snapshots"], "w") as f:
                json.dump(engine.daily_snapshots, f, indent=2)

        if not running:
            break

        elapsed = (datetime.now() - start).total_seconds()
        wait = max(0, sleep_sec - elapsed)
        logger.info("Next cycle in %.0f min", wait / 60)
        for _ in range(int(wait / 10)):
            if not running:
                break
            time.sleep(10)

    # Final save
    engine.save_trades(dirs["trades_csv"])
    with open(dirs["snapshots"], "w") as f:
        json.dump(engine.daily_snapshots, f, indent=2)
    report = engine.get_performance_report()
    (dirs["reports"] / f"report_{datetime.now().strftime('%Y-%m-%d')}.txt").write_text(report)
    logger.info("ARM %s STOPPED\n%s", timeframe, report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", required=True, choices=list(TIMEFRAME_CONFIG.keys()))
    args = parser.parse_args()
    run_arm(args.timeframe)


if __name__ == "__main__":
    main()
