#!/usr/bin/env python3
"""
Weekly Auto-Retrain — Keeps ML models fresh with new data.

Runs weekly (or on-demand):
1. Fetches latest data for all configured tickers
2. Retrains models with updated data
3. Compares new model vs existing model on holdout
4. Promotes new model ONLY if it improves F1 by > threshold
5. Logs everything for audit trail

Usage:
    python scripts/auto_retrain.py                    # Retrain all
    python scripts/auto_retrain.py --tickers RELIANCE TCS
    python scripts/auto_retrain.py --dry-run          # Compare without promoting

Weekly cron (add to crontab):
    0 6 * * 1 cd /Users/kartikgarg/FinanceBot && python scripts/auto_retrain.py >> data/retrain.log 2>&1
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.finance.analysis.data_fetcher import DataFetcher
from src.finance.cost_model import estimate_round_trip_pct
from src.ml.feature_pipeline import FeaturePipeline
from src.ml.label_generator import LabelGenerator
from src.ml.walk_forward import auto_configure_cv
from src.ml.models import TradingModelTrainer
from src.ml.hyperparameter_tuner import HyperparameterTuner
from src.ml.shap_analyzer import PatternValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("auto_retrain")

MODEL_DIR = Path("data/ml_models")
PHASE2_DIR = Path("data/ml_models/phase2")
RETRAIN_DIR = Path("data/ml_models/retrain_history")

from src.ml.ticker_config import INDIA_TICKERS_FLAT

TICKERS_CONFIG = {
    "india": {
        "tickers": INDIA_TICKERS_FLAT,  # 30 Nifty tickers
        "interval": "1h",
        "days": 59,
        "broker": "zerodha",
    },
}

PROMOTION_THRESHOLD = 0.02  # New model must improve F1 by at least 2%


def load_existing_metadata(ticker: str) -> dict | None:
    """Load the current best model's metadata."""
    safe = ticker.replace("/", "_")
    for search_dir in [PHASE2_DIR, MODEL_DIR]:
        meta_path = search_dir / f"{safe}_metadata.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text())
    return None


def retrain_ticker(ticker: str, market: str, cfg: dict, optuna_trials: int = 10,
                   dry_run: bool = False) -> dict | None:
    """Retrain a single ticker and compare with existing model."""
    safe = ticker.replace("/", "_")
    broker = cfg["broker"]

    logger.info("=" * 60)
    logger.info("RETRAINING: %s (%s)", ticker, market)
    logger.info("=" * 60)

    # Fetch latest data
    fetcher = DataFetcher()
    if market == "india":
        df = fetcher.fetch_indian_stock(ticker, cfg["interval"], cfg["days"])
    else:
        df = fetcher.fetch_us_stock(ticker, cfg["interval"], cfg["days"])

    if df is None or len(df) < 100:
        logger.warning("[%s] Insufficient data (%s bars)", ticker,
                      len(df) if df is not None else 0)
        return None

    logger.info("[%s] Data: %d bars (%s)", ticker, len(df), cfg["interval"])

    # Feature engineering
    pipeline = FeaturePipeline()
    features = pipeline.transform(df)
    features = pipeline.remove_correlated(features)

    # Labels
    cost_pct = estimate_round_trip_pct(broker)
    cost_mult = 1.0 if cfg["interval"] in ("1h", "15m", "5m") else 2.0
    label_gen = LabelGenerator({
        "round_trip_cost_pct": cost_pct,
        "cost_multiplier": cost_mult,
    })
    labels = label_gen.generate(df)

    common = features.index.intersection(labels.dropna().index)
    X, y = features.loc[common], labels.loc[common]

    if len(X) < 100:
        logger.warning("[%s] Only %d aligned samples", ticker, len(X))
        return None

    # Walk-forward CV
    cv = auto_configure_cv(len(X), min_folds=4)
    X_cv, y_cv, X_hold, y_hold = cv.get_holdout_split(X, y, 0.15)
    folds = cv.split_dataframe(X_cv, y_cv)
    class_weights = label_gen.get_class_weights(y_cv)

    # Optuna HPO (LightGBM only — XGBoost segfaults)
    label_counts = y_cv.value_counts()
    min_class_pct = label_counts.min() / len(y_cv) * 100
    best_params = {}

    if min_class_pct >= 10 and optuna_trials > 0:
        tuner = HyperparameterTuner(n_trials=optuna_trials)
        best_params = tuner.tune_lightgbm(X_cv, y_cv, cv, class_weights)

    # Train new model
    trainer = TradingModelTrainer({
        "models": ["lightgbm"],
        "lightgbm_params": best_params,
    })
    results = trainer.train_and_evaluate(folds, class_weights)

    if not results or "lightgbm" not in results:
        logger.error("[%s] Training failed", ticker)
        return None

    new_cv_f1 = results["lightgbm"].mean_f1

    # Holdout evaluation
    final_model = trainer.train_final_model("lightgbm", X_cv, y_cv, class_weights)
    from sklearn.metrics import f1_score
    preds_raw = final_model.predict(X_hold)
    preds = np.array([{0: -1, 1: 0, 2: 1}[p] for p in preds_raw])
    new_holdout_f1 = f1_score(y_hold.astype(int), preds, average="weighted", zero_division=0)

    # Compare with existing model
    existing_meta = load_existing_metadata(ticker)
    old_f1 = existing_meta.get("holdout_f1", 0) if existing_meta else 0

    improvement = new_holdout_f1 - old_f1
    should_promote = improvement > PROMOTION_THRESHOLD

    logger.info("[%s] Old F1: %.3f → New F1: %.3f (diff: %+.3f) — %s",
                ticker, old_f1, new_holdout_f1, improvement,
                "PROMOTE" if should_promote else "KEEP OLD")

    result = {
        "ticker": ticker,
        "market": market,
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(X),
        "old_f1": old_f1,
        "new_cv_f1": new_cv_f1,
        "new_holdout_f1": new_holdout_f1,
        "improvement": improvement,
        "promoted": False,
        "features": X.columns.tolist(),
    }

    if should_promote and not dry_run:
        # Save new model
        import joblib
        model_path = PHASE2_DIR / f"{safe}_lightgbm.joblib"
        PHASE2_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, model_path)

        # Save metadata
        meta = {
            "ticker": ticker,
            "market": market,
            "phase": "retrain",
            "interval": cfg["interval"],
            "n_samples": len(X),
            "n_features": len(X.columns),
            "features": X.columns.tolist(),
            "cost_pct": cost_pct,
            "best_model": "lightgbm_retrained",
            "cv_f1": new_cv_f1,
            "holdout_f1": new_holdout_f1,
            "retrained_at": datetime.now().isoformat(),
            "previous_f1": old_f1,
            "improvement": improvement,
        }
        (PHASE2_DIR / f"{safe}_metadata.json").write_text(json.dumps(meta, indent=2, default=str))

        result["promoted"] = True
        logger.info("[%s] New model PROMOTED (F1: %.3f → %.3f)", ticker, old_f1, new_holdout_f1)
    elif should_promote and dry_run:
        logger.info("[%s] WOULD promote (dry run)", ticker)
    else:
        logger.info("[%s] Keeping existing model (improvement %.3f < threshold %.3f)",
                    ticker, improvement, PROMOTION_THRESHOLD)

    return result


def main():
    parser = argparse.ArgumentParser(description="Weekly Auto-Retrain")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to retrain")
    parser.add_argument("--markets", nargs="+", default=["india", "us"])
    parser.add_argument("--optuna-trials", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Compare without promoting")
    args = parser.parse_args()

    RETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for market in args.markets:
        if market not in TICKERS_CONFIG:
            continue
        cfg = TICKERS_CONFIG[market]
        tickers = args.tickers if args.tickers else cfg["tickers"]

        for ticker in tickers:
            try:
                r = retrain_ticker(ticker, market, cfg, args.optuna_trials, args.dry_run)
                if r:
                    results.append(r)
            except Exception as e:
                logger.error("[%s] Retrain failed: %s", ticker, e)

    # Summary
    if results:
        print("\n" + "=" * 70)
        print("WEEKLY RETRAIN SUMMARY — %s" % datetime.now().strftime("%Y-%m-%d"))
        print("=" * 70)
        print(f"\n{'Ticker':<12} {'Old F1':>8} {'New F1':>8} {'Change':>8} {'Action'}")
        print("-" * 50)

        promoted = 0
        for r in results:
            action = "PROMOTED" if r["promoted"] else ("WOULD PROMOTE" if args.dry_run and r["improvement"] > PROMOTION_THRESHOLD else "kept")
            if r["promoted"]:
                promoted += 1
            print(f"  {r['ticker']:<10} {r['old_f1']:>7.3f} {r['new_holdout_f1']:>7.3f} "
                  f"{r['improvement']:>+7.3f} {action}")

        print(f"\nModels promoted: {promoted}/{len(results)}")

        # Save retrain log
        log_path = RETRAIN_DIR / f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
