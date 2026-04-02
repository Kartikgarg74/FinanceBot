#!/usr/bin/env python3
"""
Daily Paper Trading Report Generator.

Reads the live log, generates a clean daily report, archives it,
and saves to data/ml_models/daily_reports/ for review.

Usage:
    python scripts/daily_report.py              # Today's report
    python scripts/daily_report.py --date 2026-04-01
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime, date
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("daily_report")

LOG_PATH = Path("data/paper_trading/live_session.log")
REPORT_DIR = Path("data/ml_models/daily_reports")
FEEDBACK_PATH = Path("data/paper_trading/feedback_buffer.json")


def parse_log_for_date(target_date: str) -> dict:
    """Parse the live log for a specific date."""
    result = {
        "date": target_date,
        "cycles": 0,
        "tickers_scanned": 0,
        "signals": [],
        "trades": [],
        "portfolio_snapshots": [],
        "errors": [],
        "warnings": [],
    }

    if not LOG_PATH.exists():
        return result

    for line in LOG_PATH.read_text().splitlines():
        if target_date not in line:
            continue

        if "Cycle" in line and "---" in line:
            result["cycles"] += 1
        elif "SIGNAL:" in line:
            result["signals"].append(line.strip())
        elif "TRADE EXECUTED" in line or "PAPER BUY" in line or "PAPER SELL" in line:
            result["trades"].append(line.strip())
        elif "Portfolio:" in line:
            # Extract capital and positions
            match = re.search(r"capital=(\d+\.\d+).*positions=(\d+).*trades=(\d+).*drawdown=(\d+\.\d+)", line)
            if match:
                result["portfolio_snapshots"].append({
                    "time": line[:19],
                    "capital": float(match.group(1)),
                    "positions": int(match.group(2)),
                    "trades": int(match.group(3)),
                    "drawdown": float(match.group(4)),
                })
        elif "ERROR" in line:
            result["errors"].append(line.strip())
        elif "WARNING" in line:
            result["warnings"].append(line.strip())
        elif "Feature pipeline" in line:
            result["tickers_scanned"] += 1

    return result


def generate_report(data: dict) -> str:
    """Generate formatted daily report."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"  FINANCEBOT DAILY REPORT — {data['date']}")
    lines.append(f"{'='*60}")
    lines.append("")

    # Summary
    last_snap = data["portfolio_snapshots"][-1] if data["portfolio_snapshots"] else None
    capital = last_snap["capital"] if last_snap else 100000
    total_trades = last_snap["trades"] if last_snap else 0
    drawdown = last_snap["drawdown"] if last_snap else 0

    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Cycles completed:    {data['cycles']}")
    lines.append(f"  Tickers scanned:     {data['tickers_scanned']}")
    lines.append(f"  Signals generated:   {len(data['signals'])}")
    lines.append(f"  Trades executed:     {len(data['trades'])}")
    lines.append(f"  Current capital:     Rs {capital:,.2f}")
    lines.append(f"  P&L today:           Rs {capital - 100000:+,.2f}")
    lines.append(f"  Max drawdown:        {drawdown:.1f}%")
    lines.append(f"  Errors:              {len(data['errors'])}")
    lines.append("")

    # Signals
    if data["signals"]:
        lines.append("SIGNALS")
        lines.append("-" * 40)
        for sig in data["signals"]:
            lines.append(f"  {sig}")
        lines.append("")

    # Trades
    if data["trades"]:
        lines.append("TRADES")
        lines.append("-" * 40)
        for trade in data["trades"]:
            lines.append(f"  {trade}")
        lines.append("")

    # Portfolio timeline
    if data["portfolio_snapshots"]:
        lines.append("PORTFOLIO TIMELINE")
        lines.append("-" * 40)
        for snap in data["portfolio_snapshots"]:
            lines.append(f"  {snap['time']} | Capital: {snap['capital']:,.2f} | "
                        f"Pos: {snap['positions']} | Trades: {snap['trades']} | "
                        f"DD: {snap['drawdown']:.1f}%")
        lines.append("")

    # Self-learning stats
    if FEEDBACK_PATH.exists():
        try:
            feedback = json.loads(FEEDBACK_PATH.read_text())
            today_feedback = [f for f in feedback if data["date"] in f.get("timestamp", "")]
            if today_feedback:
                correct = sum(1 for f in today_feedback if f["correct"])
                lines.append("SELF-LEARNING")
                lines.append("-" * 40)
                lines.append(f"  Trades learned from today: {len(today_feedback)}")
                lines.append(f"  Prediction accuracy:       {correct/len(today_feedback):.1%}")
                lines.append(f"  Avg P&L per trade:         {sum(f['net_pnl'] for f in today_feedback)/len(today_feedback):+.2f}")
                lines.append("")
        except Exception:
            pass

    # Errors
    if data["errors"]:
        lines.append("ERRORS")
        lines.append("-" * 40)
        for err in data["errors"][:10]:
            lines.append(f"  {err}")
        lines.append("")

    # No-trade analysis
    if not data["trades"] and not data["signals"]:
        lines.append("NO-TRADE ANALYSIS")
        lines.append("-" * 40)
        lines.append("  All models returned HOLD for all tickers.")
        lines.append("  Possible reasons:")
        lines.append("  - Market was closed or low volatility")
        lines.append("  - Model confidence below threshold")
        lines.append("  - Cost-aware labels filter small moves")
        lines.append("")

    lines.append(f"{'='*60}")
    lines.append(f"  Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"{'='*60}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    data = parse_log_for_date(args.date)
    report = generate_report(data)

    # Print to stdout
    print(report)

    # Save to file
    report_path = REPORT_DIR / f"report_{args.date}.txt"
    report_path.write_text(report)
    print(f"\nSaved: {report_path}")

    # Also save as JSON for programmatic access
    json_path = REPORT_DIR / f"report_{args.date}.json"
    json_path.write_text(json.dumps(data, indent=2, default=str))


if __name__ == "__main__":
    main()
