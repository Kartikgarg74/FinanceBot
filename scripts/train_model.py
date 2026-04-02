#!/usr/bin/env python3
"""
End-to-end ML training pipeline for FinanceBot.

Usage:
    python scripts/train_model.py --ticker RELIANCE --market india --interval 15m --days 150
    python scripts/train_model.py --ticker AAPL --market us --interval 1d --days 365
    python scripts/train_model.py --ticker BTC/USDT --market crypto --interval 1h --days 180
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.finance.analysis.data_fetcher import DataFetcher
from src.finance.cost_model import estimate_round_trip_pct
from src.ml.feature_pipeline import FeaturePipeline
from src.ml.label_generator import LabelGenerator
from src.ml.walk_forward import WalkForwardCV, auto_configure_cv
from src.ml.models import TradingModelTrainer
from src.ml.shap_analyzer import SHAPAnalyzer, PatternValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_model")


def fetch_data(ticker: str, market: str, interval: str, days: int) -> pd.DataFrame:
    """Fetch OHLCV data based on market type."""
    fetcher = DataFetcher()

    if market == "india":
        df = fetcher.fetch_indian_stock(ticker, interval=interval, days=days)
    elif market == "us":
        df = fetcher.fetch_us_stock(ticker, interval=interval, days=days)
    elif market == "crypto":
        if interval in ["1d", "1h", "4h", "15m", "5m", "1m"]:
            df = fetcher.fetch_crypto_ccxt(pair=ticker, timeframe=interval, limit=min(days * 24, 1000))
        else:
            df = fetcher.fetch_crypto_yfinance(ticker.split("/")[0], interval=interval, days=days)
    else:
        raise ValueError(f"Unknown market: {market}")

    if df is None or df.empty:
        raise ValueError(f"No data fetched for {ticker} ({market}, {interval}, {days}d)")

    logger.info("Fetched %d bars for %s (%s, %s)", len(df), ticker, market, interval)
    return df


def get_broker_for_market(market: str) -> str:
    """Map market to default broker."""
    return {"india": "zerodha", "us": "alpaca", "crypto": "binance"}[market]


def main():
    parser = argparse.ArgumentParser(description="Train ML trading model")
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g., RELIANCE, AAPL, BTC/USDT)")
    parser.add_argument("--market", required=True, choices=["india", "us", "crypto"])
    parser.add_argument("--interval", default="1d", help="Candle interval (1d, 1h, 15m, 5m)")
    parser.add_argument("--days", type=int, default=365, help="Days of historical data")
    parser.add_argument("--label-method", default="fixed_threshold", choices=["fixed_threshold", "triple_barrier"])
    parser.add_argument("--models", nargs="+", default=["lightgbm", "xgboost", "random_forest"])
    parser.add_argument("--output-dir", default="data/ml_models", help="Output directory for models")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP analysis (faster)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Fetch data ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Fetching data for %s (%s, %s, %dd)",
                args.ticker, args.market, args.interval, args.days)
    logger.info("=" * 60)

    df = fetch_data(args.ticker, args.market, args.interval, args.days)

    # ── Step 2: Feature engineering ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Feature engineering")
    logger.info("=" * 60)

    pipeline = FeaturePipeline()
    features = pipeline.transform(df)
    features = pipeline.remove_correlated(features)
    logger.info("Features: %d samples x %d features", len(features), len(features.columns))

    # ── Step 3: Label generation ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Label generation")
    logger.info("=" * 60)

    broker = get_broker_for_market(args.market)
    cost_pct = estimate_round_trip_pct(broker)
    logger.info("Round-trip cost for %s: %.4f%%", broker, cost_pct * 100)

    label_gen = LabelGenerator({
        "method": args.label_method,
        "round_trip_cost_pct": cost_pct,
        "cost_multiplier": 2.0,
        "horizon": 1,
    })
    labels = label_gen.generate(df)

    # Align features and labels (they may have different lengths due to dropna)
    common_idx = features.index.intersection(labels.dropna().index)
    X = features.loc[common_idx]
    y = labels.loc[common_idx]
    logger.info("Aligned dataset: %d samples", len(X))

    if len(X) < 100:
        logger.error("Not enough samples (%d). Need at least 100. Try using a shorter interval.", len(X))
        sys.exit(1)

    # ── Step 4: Walk-forward cross-validation ───────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Walk-forward cross-validation")
    logger.info("=" * 60)

    cv = auto_configure_cv(len(X))

    # Hold out final 15% for untouched evaluation
    X_cv, y_cv, X_holdout, y_holdout = cv.get_holdout_split(X, y, holdout_pct=0.15)

    folds = cv.split_dataframe(X_cv, y_cv)
    logger.info("Generated %d walk-forward folds", len(folds))

    if len(folds) < 2:
        logger.warning("Only %d folds — results may not be statistically meaningful", len(folds))

    # ── Step 5: Model training ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Training models: %s", args.models)
    logger.info("=" * 60)

    class_weights = label_gen.get_class_weights(y_cv)
    logger.info("Class weights: %s", class_weights)

    trainer = TradingModelTrainer({"models": args.models})
    results = trainer.train_and_evaluate(folds, class_weights)

    # ── Step 6: Select best model ───────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Model comparison")
    logger.info("=" * 60)

    for name, result in results.items():
        logger.info("  %s: accuracy=%.3f±%.3f, f1=%.3f",
                     name, result.mean_accuracy, result.std_accuracy, result.mean_f1)

    best_model_name = max(results, key=lambda k: results[k].mean_f1)
    logger.info("Best model: %s (f1=%.3f)", best_model_name, results[best_model_name].mean_f1)

    # ── Step 7: Train final model and evaluate on holdout ───────────
    logger.info("=" * 60)
    logger.info("STEP 7: Final model training + holdout evaluation")
    logger.info("=" * 60)

    final_model = trainer.train_final_model(best_model_name, X_cv, y_cv, class_weights)
    preds, probs = trainer.predict(final_model, X_holdout)

    from sklearn.metrics import classification_report
    y_holdout_int = y_holdout.astype(int)
    logger.info("Holdout classification report:\n%s",
                classification_report(y_holdout_int, preds, target_names=["SELL", "HOLD", "BUY"]))

    # Save model
    model_path = output_dir / f"{args.ticker.replace('/', '_')}_{best_model_name}.joblib"
    trainer.save_model(final_model, best_model_name, model_path)

    # ── Step 8: SHAP analysis ───────────────────────────────────────
    if not args.skip_shap:
        logger.info("=" * 60)
        logger.info("STEP 8: SHAP pattern analysis")
        logger.info("=" * 60)

        shap_analyzer = SHAPAnalyzer(output_dir / "shap")
        shap_result = shap_analyzer.analyze(final_model, X_holdout, best_model_name)
        report = shap_analyzer.generate_report(shap_result, X_holdout)
        print("\n" + report)

        try:
            shap_analyzer.save_plots(shap_result, X_holdout)
        except Exception as e:
            logger.warning("Could not save SHAP plots: %s", e)

        # Feature stability across folds
        fold_importances = [r.feature_importance for r in results[best_model_name].fold_results]
        validator = PatternValidator()
        stability = validator.validate_feature_stability(fold_importances)
        logger.info("Feature stability: %s (Jaccard=%.2f, %d common features)",
                     "STABLE" if stability["stable"] else "UNSTABLE",
                     stability["mean_jaccard_similarity"],
                     stability["n_common"])
        logger.info("Common top features across all folds: %s", stability["common_top_features"])

    # ── Step 9: Save pipeline metadata ──────────────────────────────
    metadata = {
        "ticker": args.ticker,
        "market": args.market,
        "interval": args.interval,
        "days": args.days,
        "n_samples": len(X),
        "n_features": len(X.columns),
        "features": X.columns.tolist(),
        "label_method": args.label_method,
        "cost_pct": cost_pct,
        "best_model": best_model_name,
        "cv_accuracy": results[best_model_name].mean_accuracy,
        "cv_f1": results[best_model_name].mean_f1,
        "model_path": str(model_path),
        "feature_importance": results[best_model_name].feature_importance,
    }

    meta_path = output_dir / f"{args.ticker.replace('/', '_')}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Metadata saved: %s", meta_path)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("  Model: %s", model_path)
    logger.info("  CV F1: %.3f", results[best_model_name].mean_f1)
    logger.info("  Features: %d", len(X.columns))
    logger.info("  Samples: %d (CV) + %d (holdout)", len(X_cv), len(X_holdout))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
