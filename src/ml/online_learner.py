"""Online learning — the model learns from its own paper trading results.

Two learning modes:
1. **Feedback loop**: After each trade closes, the outcome (profit/loss) is used
   to adjust the model's behavior via incremental retraining.
2. **Experience replay**: Accumulates trade outcomes in a buffer, periodically
   retrains the model on recent data + trade feedback.

This makes the model SELF-LEARNING — it gets better over time by learning
from what worked and what didn't.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)


class TradeFeedback:
    """Records trade outcomes for learning."""

    def __init__(self, buffer_path: str | Path = "data/paper_trading/feedback_buffer.json"):
        self.buffer_path = Path(buffer_path)
        self.buffer_path.parent.mkdir(parents=True, exist_ok=True)
        self.buffer: list[dict] = []
        self._load_buffer()

    def record_trade(
        self,
        ticker: str,
        features: dict,
        prediction: int,
        actual_outcome: int,  # 1=profit, -1=loss, 0=breakeven
        net_pnl: float,
        confidence: float,
    ):
        """Record a completed trade's outcome for future learning."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "features": features,
            "predicted": prediction,
            "actual": actual_outcome,
            "correct": prediction == actual_outcome,
            "net_pnl": net_pnl,
            "confidence": confidence,
        }
        self.buffer.append(entry)
        self._save_buffer()

        logger.info("[FEEDBACK] %s: predicted=%d, actual=%d, correct=%s, pnl=%.2f",
                    ticker, prediction, actual_outcome, entry["correct"], net_pnl)

    def get_accuracy(self, last_n: int = 50) -> dict:
        """Get recent prediction accuracy stats."""
        recent = self.buffer[-last_n:] if len(self.buffer) > last_n else self.buffer
        if not recent:
            return {"total": 0, "correct": 0, "accuracy": 0, "avg_pnl": 0}

        correct = sum(1 for t in recent if t["correct"])
        avg_pnl = np.mean([t["net_pnl"] for t in recent])
        profitable = sum(1 for t in recent if t["net_pnl"] > 0)

        return {
            "total": len(recent),
            "correct": correct,
            "accuracy": correct / len(recent),
            "profitable": profitable,
            "win_rate": profitable / len(recent),
            "avg_pnl": float(avg_pnl),
            "total_pnl": float(sum(t["net_pnl"] for t in recent)),
        }

    def get_ticker_stats(self) -> dict[str, dict]:
        """Get per-ticker learning stats."""
        stats = {}
        for entry in self.buffer:
            ticker = entry["ticker"]
            if ticker not in stats:
                stats[ticker] = {"trades": 0, "correct": 0, "total_pnl": 0, "wins": 0}
            stats[ticker]["trades"] += 1
            if entry["correct"]:
                stats[ticker]["correct"] += 1
            if entry["net_pnl"] > 0:
                stats[ticker]["wins"] += 1
            stats[ticker]["total_pnl"] += entry["net_pnl"]

        for ticker in stats:
            s = stats[ticker]
            s["accuracy"] = s["correct"] / s["trades"] if s["trades"] > 0 else 0
            s["win_rate"] = s["wins"] / s["trades"] if s["trades"] > 0 else 0

        return stats

    def _load_buffer(self):
        if self.buffer_path.exists():
            self.buffer = json.loads(self.buffer_path.read_text())
            logger.info("Loaded %d feedback entries", len(self.buffer))

    def _save_buffer(self):
        self.buffer_path.write_text(json.dumps(self.buffer, indent=2, default=str))


class OnlineLearner:
    """Incrementally improves models from paper trading feedback.

    Strategy: After accumulating N trade outcomes, retrain the model
    using a mix of original training data + recent trade feedback.
    Trades that were profitable get reinforced; losing trades get corrected.
    """

    def __init__(
        self,
        model_dir: str | Path = "data/ml_models/phase2",
        feedback_buffer: TradeFeedback | None = None,
        retrain_threshold: int = 20,  # Retrain after N new trades
        interval: str = "1h",
        days: int = 59,
    ):
        self.model_dir = Path(model_dir)
        self.feedback = feedback_buffer or TradeFeedback()
        self.retrain_threshold = retrain_threshold
        self.interval = interval
        self.days = days
        self._trades_since_retrain = 0

    def on_trade_closed(
        self,
        ticker: str,
        features_dict: dict,
        predicted_action: int,
        net_pnl: float,
        confidence: float,
    ):
        """Called when a paper trade closes. Records feedback and may trigger retrain."""
        # Determine actual outcome from P&L
        if net_pnl > 0:
            actual = predicted_action  # Prediction was right direction
        elif net_pnl < 0:
            actual = -predicted_action  # Prediction was wrong direction
        else:
            actual = 0  # Breakeven

        self.feedback.record_trade(
            ticker=ticker,
            features=features_dict,
            prediction=predicted_action,
            actual_outcome=actual,
            net_pnl=net_pnl,
            confidence=confidence,
        )

        self._trades_since_retrain += 1

        # Check if we should retrain
        if self._trades_since_retrain >= self.retrain_threshold:
            logger.info("Retrain threshold reached (%d trades). Triggering incremental retrain.",
                       self._trades_since_retrain)
            self.incremental_retrain(ticker)
            self._trades_since_retrain = 0

    def incremental_retrain(self, ticker: str):
        """Retrain a ticker's model using feedback-weighted data."""
        safe = ticker.replace("/", "_")
        model_path = self.model_dir / f"{safe}_lightgbm.joblib"
        meta_path = self.model_dir / f"{safe}_metadata.json"

        if not model_path.exists() or not meta_path.exists():
            logger.warning("No model found for %s — skipping retrain", ticker)
            return

        try:
            import lightgbm as lgb
            from src.finance.analysis.data_fetcher import DataFetcher
            from src.ml.feature_pipeline import FeaturePipeline
            from src.ml.label_generator import LabelGenerator
            from src.finance.cost_model import estimate_round_trip_pct

            meta = json.loads(meta_path.read_text())

            # Fetch fresh data
            fetcher = DataFetcher()
            df = fetcher.fetch_indian_stock(ticker, self.interval, self.days)
            if df is None or len(df) < 100:
                return

            # Build features + labels
            pipeline = FeaturePipeline()
            features = pipeline.transform(df)
            features = pipeline.remove_correlated(features)

            cost = estimate_round_trip_pct("zerodha")
            label_gen = LabelGenerator({"round_trip_cost_pct": cost, "cost_multiplier": 1.0})
            labels = label_gen.generate(df)

            common = features.index.intersection(labels.dropna().index)
            X = features.loc[common]
            y = labels.loc[common]

            trained_feats = meta.get("features", [])
            if trained_feats:
                available = [f for f in trained_feats if f in X.columns]
                X = X[available]

            # Map labels
            y_mapped = y.map({-1: 0, 0: 1, 1: 2}).astype(int)

            # Build sample weights from feedback
            # Recent trades that were WRONG get higher weight (model should learn from mistakes)
            sample_weights = np.ones(len(X))
            ticker_feedback = [e for e in self.feedback.buffer if e["ticker"] == ticker]
            if ticker_feedback:
                wrong_count = sum(1 for e in ticker_feedback[-20:] if not e["correct"])
                right_count = sum(1 for e in ticker_feedback[-20:] if e["correct"])
                # Boost weight on recent data if model has been wrong
                error_rate = wrong_count / max(wrong_count + right_count, 1)
                # Give more weight to recent bars (last 20%)
                recent_start = int(len(X) * 0.8)
                sample_weights[recent_start:] *= (1 + error_rate)
                logger.info("[%s] Feedback: %d right, %d wrong (error_rate=%.2f). Boosting recent data.",
                           ticker, right_count, wrong_count, error_rate)

            # Retrain with warm start
            model = lgb.LGBMClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.01,
                subsample=0.7, colsample_bytree=0.7, min_child_samples=20,
                verbose=-1, random_state=42, n_jobs=1,
            )
            model.fit(X, y_mapped, sample_weight=sample_weights)

            # Save updated model
            joblib.dump(model, model_path)

            # Update metadata
            meta["last_retrain"] = datetime.now().isoformat()
            meta["retrain_reason"] = "online_learning"
            meta["feedback_trades"] = len(ticker_feedback)
            meta_path.write_text(json.dumps(meta, indent=2, default=str))

            logger.info("[%s] Model retrained with %d samples (feedback: %d trades)",
                       ticker, len(X), len(ticker_feedback))

        except Exception as e:
            logger.error("[%s] Incremental retrain failed: %s", ticker, e)

    def get_learning_report(self) -> str:
        """Generate a report on the model's learning progress."""
        stats = self.feedback.get_accuracy()
        ticker_stats = self.feedback.get_ticker_stats()

        lines = [
            "Self-Learning Report",
            "=" * 50,
            f"Total feedback trades: {stats['total']}",
            f"Prediction accuracy:   {stats['accuracy']:.1%}",
            f"Win rate (profitable): {stats.get('win_rate', 0):.1%}",
            f"Average P&L per trade: {stats['avg_pnl']:+.2f}",
            f"Total P&L:             {stats.get('total_pnl', 0):+.2f}",
            f"Trades until retrain:  {self.retrain_threshold - self._trades_since_retrain}",
            "",
        ]

        if ticker_stats:
            lines.append("Per-Ticker Learning:")
            lines.append(f"{'Ticker':<14} {'Trades':>7} {'Accuracy':>9} {'Win%':>6} {'P&L':>10}")
            lines.append("-" * 50)
            for ticker, s in sorted(ticker_stats.items(), key=lambda x: -x[1]["total_pnl"]):
                lines.append(f"{ticker:<14} {s['trades']:>7} {s['accuracy']:>8.1%} "
                            f"{s['win_rate']:>5.1%} {s['total_pnl']:>+9.2f}")

        return "\n".join(lines)
