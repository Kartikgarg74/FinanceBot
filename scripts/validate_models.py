#!/usr/bin/env python3
"""
Comprehensive model validation pipeline.

1. Cross-checks tickers against major indices (Nifty 50, S&P 500)
2. Runs ML backtests with exact broker fees
3. Compares ML vs Buy-and-Hold vs Rule-Based
4. Runs statistical validation (t-test, Monte Carlo, Deflated Sharpe)
5. Generates a full validation report

Usage:
    python scripts/validate_models.py
    python scripts/validate_models.py --markets india us
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate")

# ── Major Index Constituents (top holdings for cross-checking) ──────
NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
    "HCLTECH", "TITAN", "SUNPHARMA", "WIPRO", "ULTRACEMCO",
    "NESTLEIND", "TECHM", "POWERGRID", "NTPC", "TATAMOTORS",
    "M&M", "ADANIENT", "ADANIPORTS", "JSWSTEEL", "TATASTEEL",
    "BAJAJFINSV", "ONGC", "COALINDIA", "BPCL", "GRASIM",
    "DIVISLAB", "DRREDDY", "CIPLA", "APOLLOHOSP", "EICHERMOT",
    "HEROMOTOCO", "TATACONSUM", "SBILIFE", "BRITANNIA", "INDUSINDBK",
    "HDFCLIFE", "BAJAJ-AUTO", "HINDALCO", "UPL", "SHRIRAMFIN",
]

SP500_TOP = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "BRK-B", "UNH", "JNJ",
    "V", "XOM", "JPM", "PG", "MA",
    "HD", "CVX", "MRK", "ABBV", "LLY",
    "AVGO", "PEP", "KO", "COST", "TMO",
    "MCD", "WMT", "CSCO", "ACN", "ABT",
]

CRYPTO_MAJORS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT",
]

MODEL_DIR = Path("data/ml_models")


def cross_check_tickers():
    """Cross-check trained tickers against major indices."""
    print("\n" + "=" * 70)
    print("TICKER VALIDATION — Cross-check against Major Indices")
    print("=" * 70)

    trained = []
    for p in MODEL_DIR.glob("*_metadata.json"):
        meta = json.loads(p.read_text())
        trained.append({
            "ticker": meta["ticker"],
            "market": meta["market"],
            "model": meta["best_model"],
            "holdout_f1": meta.get("holdout_f1", 0),
        })

    if not trained:
        print("No trained models found.")
        return trained

    print(f"\nTrained models: {len(trained)}")
    print(f"\n{'Ticker':<15} {'Market':<8} {'In Index?':<25} {'F1':<8}")
    print("-" * 60)

    for t in trained:
        ticker = t["ticker"]
        market = t["market"]
        if market == "india":
            in_index = "NIFTY 50" if ticker in NIFTY_50 else "NOT in Nifty 50"
        elif market == "us":
            in_index = "S&P 500 Top 30" if ticker in SP500_TOP else "NOT in S&P 500 Top"
        elif market == "crypto":
            in_index = "Top 10 Crypto" if ticker in CRYPTO_MAJORS else "NOT in Top 10"
        else:
            in_index = "Unknown"

        status = "✓" if "NOT" not in in_index else "?"
        print(f"{status} {ticker:<13} {market:<8} {in_index:<25} {t['holdout_f1']:.3f}")

    # Show which top index stocks we HAVEN'T trained on
    trained_india = {t["ticker"] for t in trained if t["market"] == "india"}
    trained_us = {t["ticker"] for t in trained if t["market"] == "us"}

    missing_india = [t for t in NIFTY_50[:10] if t not in trained_india]
    missing_us = [t for t in SP500_TOP[:10] if t not in trained_us]

    if missing_india:
        print(f"\nNifty 50 top stocks NOT yet trained: {', '.join(missing_india)}")
    if missing_us:
        print(f"S&P 500 top stocks NOT yet trained: {', '.join(missing_us)}")

    return trained


def run_buy_and_hold(df: pd.DataFrame, initial_capital: float,
                     broker: str, trade_type: str) -> dict:
    """Calculate buy-and-hold returns with exact entry/exit costs."""
    if df is None or df.empty:
        return {"net_pnl": 0, "pnl_pct": 0, "total_fees": 0}

    cost_model = get_cost_model(broker, trade_type=trade_type, slippage_bps=10)

    entry_price = float(df["Close"].iloc[0])
    exit_price = float(df["Close"].iloc[-1])

    # Buy at first bar
    qty = int(initial_capital / entry_price)
    if qty == 0:
        return {"net_pnl": 0, "pnl_pct": 0, "total_fees": 0}

    buy_value = qty * entry_price
    buy_cost = cost_model.calculate(buy_value, "buy")

    # Sell at last bar
    sell_value = qty * exit_price
    sell_cost = cost_model.calculate(sell_value, "sell")

    gross_pnl = (exit_price - entry_price) * qty
    total_fees = buy_cost.total + sell_cost.total
    net_pnl = gross_pnl - total_fees

    return {
        "net_pnl": round(net_pnl, 2),
        "pnl_pct": round(net_pnl / initial_capital * 100, 2),
        "total_fees": round(total_fees, 2),
        "gross_pnl": round(gross_pnl, 2),
    }


def validate_single_ticker(ticker: str, market: str, model_path: Path,
                           initial_capital: float = 100000) -> dict | None:
    """Full validation for a single ticker."""
    broker_map = {"india": "zerodha", "us": "alpaca", "crypto": "binance"}
    trade_type_map = {"india": "intraday", "us": "stock", "crypto": "spot"}
    days_map = {"india": 365, "us": 730, "crypto": 180}

    broker = broker_map[market]
    trade_type = trade_type_map[market]

    # Fetch data
    fetcher = DataFetcher()
    if market == "india":
        df = fetcher.fetch_indian_stock(ticker, "1d", days_map[market])
    elif market == "us":
        df = fetcher.fetch_us_stock(ticker, "1d", days_map[market])
    else:
        df = fetcher.fetch_crypto_ccxt(ticker, "1h", min(days_map[market] * 24, 1000))

    if df is None or df.empty:
        return None

    backtester = Backtester()

    # 1. ML Backtest
    ml_result = backtester.run_ml(
        df=df, ticker=ticker, model_path=model_path,
        initial_capital=initial_capital, broker=broker,
        trade_type=trade_type, slippage_bps=10,
    )

    # 2. Rule-based backtest
    rule_result = backtester.run(
        df=df, ticker=ticker, initial_capital=initial_capital,
        broker=broker, trade_type=trade_type, slippage_bps=10,
    )

    # 3. Buy-and-hold
    bnh = run_buy_and_hold(df, initial_capital, broker, trade_type)

    # 4. Statistical validation on ML trades
    validation = {}
    if ml_result.trades:
        returns = np.array([t["net_pnl"] / initial_capital for t in ml_result.trades])
        validator = PatternValidator()
        validation = validator.validate_returns(returns)

    cost_pct = estimate_round_trip_pct(broker, trade_value=initial_capital / 10)

    return {
        "ticker": ticker,
        "market": market,
        "bars": len(df),
        "cost_pct": round(cost_pct * 100, 3),
        "ml": {
            "net_pnl": ml_result.net_pnl,
            "pnl_pct": ml_result.pnl_pct,
            "trades": ml_result.total_trades,
            "win_rate": ml_result.win_rate,
            "sharpe": ml_result.sharpe_ratio,
            "sortino": ml_result.sortino_ratio,
            "max_dd": ml_result.max_drawdown_pct,
            "profit_factor": ml_result.profit_factor,
            "fees": ml_result.total_fees,
        },
        "rule_based": {
            "net_pnl": rule_result.net_pnl,
            "pnl_pct": rule_result.pnl_pct,
            "trades": rule_result.total_trades,
            "win_rate": rule_result.win_rate,
            "sharpe": rule_result.sharpe_ratio,
            "fees": rule_result.total_fees,
        },
        "buy_and_hold": bnh,
        "validation": {
            "t_test_p": validation.get("t_test", {}).get("p_value", 1.0),
            "t_test_sig": validation.get("t_test", {}).get("significant", False),
            "mc_p": validation.get("monte_carlo", {}).get("p_value", 1.0),
            "mc_sig": validation.get("monte_carlo", {}).get("significant", False),
            "sharpe": validation.get("sharpe_ratio", 0),
            "profit_factor": validation.get("profit_factor", 0),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", nargs="+", default=["india", "us"])
    parser.add_argument("--capital", type=float, default=100000)
    args = parser.parse_args()

    # Step 1: Cross-check tickers
    trained = cross_check_tickers()
    if not trained:
        return

    # Step 2: Run validation per ticker
    print("\n" + "=" * 70)
    print("FULL VALIDATION — ML vs Rule-Based vs Buy-and-Hold (with fees)")
    print("=" * 70)

    results = []
    for t in trained:
        if t["market"] not in args.markets:
            continue

        # Find model file
        safe = t["ticker"].replace("/", "_")
        model_files = list(MODEL_DIR.glob(f"{safe}_*.joblib"))
        if not model_files:
            logger.warning("No model file for %s", t["ticker"])
            continue

        model_path = model_files[0]
        logger.info("Validating %s with %s", t["ticker"], model_path.name)

        result = validate_single_ticker(
            t["ticker"], t["market"], model_path, args.capital)
        if result:
            results.append(result)

    if not results:
        print("No validation results.")
        return

    # Step 3: Print comparison table
    print(f"\n{'Ticker':<12} {'Mkt':<6} {'ML P&L%':>10} {'Rule P&L%':>10} "
          f"{'B&H P&L%':>10} {'ML Fees':>10} {'ML Trades':>10} "
          f"{'Win%':>8} {'Sharpe':>8} {'t-test':>8}")
    print("-" * 105)

    for r in sorted(results, key=lambda x: -x["ml"]["pnl_pct"]):
        t_sig = "SIG" if r["validation"]["t_test_sig"] else "n.s."
        print(f"{r['ticker']:<12} {r['market']:<6} "
              f"{r['ml']['pnl_pct']:>+9.1f}% "
              f"{r['rule_based']['pnl_pct']:>+9.1f}% "
              f"{r['buy_and_hold']['pnl_pct']:>+9.1f}% "
              f"{r['ml']['fees']:>10,.0f} "
              f"{r['ml']['trades']:>10} "
              f"{r['ml']['win_rate']:>7.1f}% "
              f"{r['ml']['sharpe']:>7.2f} "
              f"{t_sig:>8}")

    # Step 4: Summary statistics
    ml_profits = [r for r in results if r["ml"]["pnl_pct"] > 0]
    ml_beats_bnh = [r for r in results
                    if r["ml"]["pnl_pct"] > r["buy_and_hold"]["pnl_pct"]]
    sig_results = [r for r in results if r["validation"]["t_test_sig"]]

    print(f"\n{'=' * 70}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total tickers validated:        {len(results)}")
    print(f"ML profitable (net of fees):    {len(ml_profits)}/{len(results)}")
    print(f"ML beats Buy-and-Hold:          {len(ml_beats_bnh)}/{len(results)}")
    print(f"Statistically significant:      {len(sig_results)}/{len(results)}")

    total_ml_fees = sum(r["ml"]["fees"] for r in results)
    total_ml_pnl = sum(r["ml"]["net_pnl"] for r in results)
    total_bnh_pnl = sum(r["buy_and_hold"]["net_pnl"] for r in results)

    print(f"\nAggregate ML P&L:               {total_ml_pnl:+,.2f}")
    print(f"Aggregate Buy-and-Hold P&L:     {total_bnh_pnl:+,.2f}")
    print(f"Total ML fees paid:             {total_ml_fees:,.2f}")

    if total_ml_pnl > total_bnh_pnl:
        print(f"\n>>> ML OUTPERFORMS Buy-and-Hold by {total_ml_pnl - total_bnh_pnl:+,.2f}")
    else:
        print(f"\n>>> Buy-and-Hold OUTPERFORMS ML by {total_bnh_pnl - total_ml_pnl:+,.2f}")
        print("    (Expected with daily data — need higher-frequency data for edge)")

    # Step 5: Fee breakdown by broker
    print(f"\n{'=' * 70}")
    print("FEE ANALYSIS")
    print(f"{'=' * 70}")
    for market in set(r["market"] for r in results):
        market_results = [r for r in results if r["market"] == market]
        avg_cost = np.mean([r["cost_pct"] for r in market_results])
        total_fees = sum(r["ml"]["fees"] for r in market_results)
        total_gross = sum(r["ml"].get("net_pnl", 0) + r["ml"]["fees"] for r in market_results)
        fee_drag = (total_fees / total_gross * 100) if total_gross > 0 else 0
        print(f"  {market.upper()}: avg round-trip cost={avg_cost:.3f}%, "
              f"total fees={total_fees:,.0f}, fee drag={fee_drag:.1f}% of gross")

    # Save full report
    report_path = MODEL_DIR / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull report saved: {report_path}")


if __name__ == "__main__":
    main()
