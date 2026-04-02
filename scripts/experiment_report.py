#!/usr/bin/env python3
"""Cross-timeframe comparison report for the experiment."""

import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.experiment_config import TIMEFRAME_CONFIG, EXPERIMENT_DIR, get_arm_dirs


def get_arm_stats(tf: str, target_date: str) -> dict:
    """Extract stats for one experiment arm."""
    dirs = get_arm_dirs(tf)
    log_path = dirs["log"]
    stats = {"tf": tf, "cycles": 0, "signals": 0, "trades": 0,
             "capital": 100000, "drawdown": 0, "errors": 0}

    if not log_path.exists():
        return stats

    for line in log_path.read_text().splitlines():
        if target_date not in line:
            continue
        if "Cycle" in line and "---" in line:
            stats["cycles"] += 1
        if "SIGNAL:" in line:
            stats["signals"] += 1
        if "TRADE:" in line:
            stats["trades"] += 1
        if "ERROR" in line:
            stats["errors"] += 1
        match = re.search(r"capital=(\d+\.?\d*)", line)
        if match:
            stats["capital"] = float(match.group(1))
        match = re.search(r"dd=(\d+\.?\d*)%", line)
        if match:
            stats["drawdown"] = float(match.group(1))

    stats["pnl"] = stats["capital"] - 100000
    stats["return_pct"] = stats["pnl"] / 1000  # percentage of 100k

    # Feedback stats
    fb_path = dirs["feedback"]
    if fb_path.exists():
        try:
            fb = json.loads(fb_path.read_text())
            today_fb = [f for f in fb if target_date in f.get("timestamp", "")]
            stats["feedback_trades"] = len(today_fb)
            if today_fb:
                stats["accuracy"] = sum(1 for f in today_fb if f["correct"]) / len(today_fb) * 100
        except Exception:
            pass

    return stats


def main():
    target = date.today().isoformat()
    report_dir = EXPERIMENT_DIR / "comparison"
    report_dir.mkdir(parents=True, exist_ok=True)

    arms = {}
    for tf in TIMEFRAME_CONFIG:
        arms[tf] = get_arm_stats(tf, target)

    # Print comparison
    print(f"\n{'='*80}")
    print(f"  MULTI-TIMEFRAME EXPERIMENT — Daily Comparison ({target})")
    print(f"{'='*80}\n")

    header = f"{'Metric':<20}"
    for tf in TIMEFRAME_CONFIG:
        header += f"{tf:>12}"
    print(header)
    print("-" * (20 + 12 * len(TIMEFRAME_CONFIG)))

    metrics = [
        ("Cycles", "cycles"),
        ("Signals", "signals"),
        ("Trades", "trades"),
        ("Capital", "capital"),
        ("P&L", "pnl"),
        ("Return %", "return_pct"),
        ("Max Drawdown %", "drawdown"),
        ("Errors", "errors"),
    ]

    for label, key in metrics:
        row = f"{label:<20}"
        for tf in TIMEFRAME_CONFIG:
            val = arms[tf].get(key, 0)
            if key == "capital":
                row += f"{val:>11,.0f}"
            elif key == "pnl":
                row += f"{val:>+11,.0f}"
            elif key in ("return_pct", "drawdown"):
                row += f"{val:>10.1f}%"
            else:
                row += f"{val:>12}"
        print(row)

    # Find leader
    leader = max(arms.items(), key=lambda x: x[1]["pnl"])
    print(f"\nLEADER: {leader[0]} (P&L: {leader[1]['pnl']:+,.0f})")

    # Save
    report_path = report_dir / f"daily_comparison_{target}.json"
    report_path.write_text(json.dumps(arms, indent=2, default=str))
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
