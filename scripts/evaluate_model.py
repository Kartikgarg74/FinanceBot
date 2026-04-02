#!/usr/bin/env python3
"""
Evaluate a trained ML model with realistic backtesting.

Runs the model on historical data with full transaction cost modeling,
generates performance metrics, and validates patterns statistically.

Usage:
    python scripts/evaluate_model.py --ticker RELIANCE --market india --model data/ml_models/RELIANCE_xgboost.joblib
    python scripts/evaluate_model.py --ticker AAPL --market us --model data/ml_models/AAPL_xgboost.joblib --capital 10000
"""

import argparse
import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.finance.analysis.data_fetcher import DataFetcher
from src.finance.backtester import Backtester
from src.finance.cost_model import get_cost_model, estimate_round_trip_pct
from src.ml.shap_analyzer import PatternValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_model")

BROKER_MAP = {"india": "zerodha", "us": "alpaca", "crypto": "binance"}
TRADE_TYPE_MAP = {"india": "intraday", "us": "stock", "crypto": "spot"}


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained ML model with realistic backtesting")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--market", required=True, choices=["india", "us", "crypto"])
    parser.add_argument("--model", required=True, help="Path to trained model .joblib")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--risk-pct", type=float, default=2.0)
    parser.add_argument("--slippage-bps", type=float, default=10)
    parser.add_argument("--output-dir", default="data/evaluations")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    broker = BROKER_MAP[args.market]
    trade_type = TRADE_TYPE_MAP[args.market]

    # ── Fetch data ──────────────────────────────────────────────────
    logger.info("Fetching data for %s (%s, %s, %dd)", args.ticker, args.market, args.interval, args.days)
    fetcher = DataFetcher()
    if args.market == "india":
        df = fetcher.fetch_indian_stock(args.ticker, args.interval, args.days)
    elif args.market == "us":
        df = fetcher.fetch_us_stock(args.ticker, args.interval, args.days)
    else:
        df = fetcher.fetch_crypto_ccxt(args.ticker, args.interval, min(args.days * 24, 1000))

    if df is None or df.empty:
        logger.error("No data for %s", args.ticker)
        sys.exit(1)

    logger.info("Data: %d bars", len(df))

    # ── Run ML backtest ─────────────────────────────────────────────
    logger.info("Running ML backtest with %s cost model (slippage=%d bps)", broker, args.slippage_bps)

    backtester = Backtester()
    ml_result = backtester.run_ml(
        df=df,
        ticker=args.ticker,
        model_path=args.model,
        initial_capital=args.capital,
        risk_per_trade_pct=args.risk_pct,
        broker=broker,
        trade_type=trade_type,
        slippage_bps=args.slippage_bps,
    )

    # ── Run rule-based backtest for comparison ──────────────────────
    logger.info("Running rule-based backtest for comparison")
    rule_result = backtester.run(
        df=df,
        ticker=args.ticker,
        initial_capital=args.capital,
        risk_per_trade_pct=args.risk_pct,
        broker=broker,
        trade_type=trade_type,
        slippage_bps=args.slippage_bps,
    )

    # ── Print comparison ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BACKTEST COMPARISON: ML vs Rule-Based")
    print("=" * 70)

    cost_pct = estimate_round_trip_pct(broker, trade_value=args.capital / 10)
    print(f"\nBroker: {broker} | Round-trip cost: {cost_pct*100:.3f}% | Slippage: {args.slippage_bps} bps\n")

    metrics = [
        ("Net P&L", f"{ml_result.net_pnl:+,.2f}", f"{rule_result.net_pnl:+,.2f}"),
        ("Net P&L %", f"{ml_result.pnl_pct:+.1f}%", f"{rule_result.pnl_pct:+.1f}%"),
        ("Total Trades", str(ml_result.total_trades), str(rule_result.total_trades)),
        ("Win Rate", f"{ml_result.win_rate:.1f}%", f"{rule_result.win_rate:.1f}%"),
        ("Profit Factor", f"{ml_result.profit_factor:.2f}", f"{rule_result.profit_factor:.2f}"),
        ("Sharpe Ratio", f"{ml_result.sharpe_ratio:.2f}", f"{rule_result.sharpe_ratio:.2f}"),
        ("Sortino Ratio", f"{ml_result.sortino_ratio:.2f}", f"{rule_result.sortino_ratio:.2f}"),
        ("Max Drawdown", f"{ml_result.max_drawdown_pct:.1f}%", f"{rule_result.max_drawdown_pct:.1f}%"),
        ("Total Fees", f"{ml_result.total_fees:,.2f}", f"{rule_result.total_fees:,.2f}"),
        ("Gross P&L", f"{ml_result.gross_pnl:+,.2f}", f"{rule_result.gross_pnl:+,.2f}"),
    ]

    print(f"{'Metric':<20} {'ML Model':<20} {'Rule-Based':<20}")
    print("-" * 60)
    for name, ml_val, rule_val in metrics:
        print(f"{name:<20} {ml_val:<20} {rule_val:<20}")

    # ── Statistical validation ──────────────────────────────────────
    if ml_result.trades:
        print("\n" + "=" * 70)
        print("STATISTICAL VALIDATION (ML Model)")
        print("=" * 70)

        returns = np.array([t["net_pnl"] / args.capital for t in ml_result.trades])
        validator = PatternValidator()
        validation = validator.validate_returns(returns)

        print(f"\nT-test p-value:     {validation['t_test']['p_value']:.4f} "
              f"({'SIGNIFICANT' if validation['t_test']['significant'] else 'NOT significant'})")
        print(f"Monte Carlo p-value: {validation['monte_carlo']['p_value']:.4f} "
              f"({'SIGNIFICANT' if validation['monte_carlo']['significant'] else 'NOT significant'})")
        print(f"Sharpe Ratio:        {validation['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:        {validation['max_drawdown']:.1%}")
        print(f"Profit Factor:       {validation['profit_factor']:.2f}")
        print(f"Win Rate:            {validation['win_rate']:.1%}")
        print(f"\nVerdict:\n{validation['verdict']}")
    else:
        print("\nNo trades generated by ML model — cannot validate.")

    # ── Save results ────────────────────────────────────────────────
    result_data = {
        "ticker": args.ticker,
        "market": args.market,
        "broker": broker,
        "interval": args.interval,
        "model_path": args.model,
        "ml_result": {
            "net_pnl": ml_result.net_pnl,
            "pnl_pct": ml_result.pnl_pct,
            "total_trades": ml_result.total_trades,
            "win_rate": ml_result.win_rate,
            "sharpe": ml_result.sharpe_ratio,
            "sortino": ml_result.sortino_ratio,
            "max_drawdown": ml_result.max_drawdown_pct,
            "profit_factor": ml_result.profit_factor,
            "total_fees": ml_result.total_fees,
        },
        "rule_result": {
            "net_pnl": rule_result.net_pnl,
            "pnl_pct": rule_result.pnl_pct,
            "total_trades": rule_result.total_trades,
            "win_rate": rule_result.win_rate,
            "sharpe": rule_result.sharpe_ratio,
            "sortino": rule_result.sortino_ratio,
            "max_drawdown": rule_result.max_drawdown_pct,
            "profit_factor": rule_result.profit_factor,
            "total_fees": rule_result.total_fees,
        },
    }

    result_path = output_dir / f"{args.ticker.replace('/', '_')}_evaluation.json"
    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2, default=str)
    logger.info("Results saved: %s", result_path)


if __name__ == "__main__":
    main()
