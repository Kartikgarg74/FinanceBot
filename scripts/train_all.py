#!/usr/bin/env python3
"""
Comprehensive training pipeline — trains and tunes models across all markets.

Runs Optuna HPO, trains on multiple tickers, evaluates with costs,
and selects the best model per ticker.

Usage:
    python scripts/train_all.py
    python scripts/train_all.py --optuna-trials 30
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
from src.finance.cost_model import estimate_round_trip_pct, get_cost_model
from src.ml.feature_pipeline import FeaturePipeline
from src.ml.label_generator import LabelGenerator
from src.ml.walk_forward import WalkForwardCV, auto_configure_cv
from src.ml.models import TradingModelTrainer
from src.ml.hyperparameter_tuner import HyperparameterTuner
from src.ml.shap_analyzer import SHAPAnalyzer, PatternValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_all")

# ── Configuration ───────────────────────────────────────────────────
from src.ml.ticker_config import INDIA_TICKERS_FLAT

TICKERS = {
    "india": {
        "tickers": INDIA_TICKERS_FLAT,  # 30 Nifty tickers across all sectors
        "interval": "1d",
        "days": 365,
        "broker": "zerodha",
        "trade_type": "intraday",
    },
}

BROKER_MAP = {"india": "zerodha", "us": "alpaca", "crypto": "binance"}
OUTPUT_DIR = Path("data/ml_models")


def fetch_data(ticker: str, market: str, interval: str, days: int) -> pd.DataFrame | None:
    """Fetch OHLCV data."""
    fetcher = DataFetcher()
    try:
        if market == "india":
            return fetcher.fetch_indian_stock(ticker, interval=interval, days=days)
        elif market == "us":
            return fetcher.fetch_us_stock(ticker, interval=interval, days=days)
        elif market == "crypto":
            return fetcher.fetch_crypto_ccxt(pair=ticker, timeframe=interval, limit=min(days * 24, 1000))
    except Exception as e:
        logger.error("Failed to fetch %s: %s", ticker, e)
    return None


def train_single_ticker(
    ticker: str,
    market: str,
    df: pd.DataFrame,
    broker: str,
    optuna_trials: int = 30,
    label_method: str = "fixed_threshold",
) -> dict | None:
    """Full training pipeline for a single ticker with HPO."""
    safe_name = ticker.replace("/", "_")
    logger.info("=" * 60)
    logger.info("TRAINING: %s (%s) — %d bars", ticker, market, len(df))
    logger.info("=" * 60)

    # ── Features ────────────────────────────────────────────────────
    pipeline = FeaturePipeline()
    try:
        features = pipeline.transform(df)
        features = pipeline.remove_correlated(features)
    except ValueError as e:
        logger.error("Feature pipeline failed for %s: %s", ticker, e)
        return None

    if len(features) < 100:
        logger.warning("Only %d samples for %s — skipping (need 100+)", len(features), ticker)
        return None

    # ── Labels ──────────────────────────────────────────────────────
    cost_pct = estimate_round_trip_pct(broker)
    label_gen = LabelGenerator({
        "method": label_method,
        "round_trip_cost_pct": cost_pct,
        "cost_multiplier": 2.0,
        "horizon": 1,
    })
    labels = label_gen.generate(df)

    common_idx = features.index.intersection(labels.dropna().index)
    X = features.loc[common_idx]
    y = labels.loc[common_idx]

    if len(X) < 100:
        logger.warning("Only %d aligned samples for %s — skipping", len(X), ticker)
        return None

    # ── Walk-forward CV ─────────────────────────────────────────────
    cv = auto_configure_cv(len(X))
    X_cv, y_cv, X_holdout, y_holdout = cv.get_holdout_split(X, y, holdout_pct=0.15)
    folds = cv.split_dataframe(X_cv, y_cv)

    if len(folds) < 2:
        logger.warning("Only %d folds for %s — not enough for reliable CV", len(folds), ticker)

    class_weights = label_gen.get_class_weights(y_cv)

    # ── Optuna HPO ──────────────────────────────────────────────────
    logger.info("[%s] Running Optuna HPO (%d trials each)...", ticker, optuna_trials)
    tuner = HyperparameterTuner(n_trials=optuna_trials)

    best_lgb_params = tuner.tune_lightgbm(X_cv, y_cv, cv, class_weights)
    best_xgb_params = tuner.tune_xgboost(X_cv, y_cv, cv, class_weights)

    # ── Train with tuned params ─────────────────────────────────────
    logger.info("[%s] Training with tuned hyperparameters...", ticker)
    trainer = TradingModelTrainer({
        "models": ["lightgbm", "xgboost"],
        "lightgbm_params": best_lgb_params,
        "xgboost_params": best_xgb_params,
    })
    results = trainer.train_and_evaluate(folds, class_weights)

    # Also train with defaults for comparison
    default_trainer = TradingModelTrainer({"models": ["lightgbm", "xgboost", "random_forest"]})
    default_results = default_trainer.train_and_evaluate(folds, class_weights)

    # Merge — pick best F1 per model across tuned vs default
    all_results = {}
    for name in set(list(results.keys()) + list(default_results.keys())):
        tuned = results.get(name)
        default = default_results.get(name)
        if tuned and default:
            all_results[name + "_tuned"] = tuned
            all_results[name + "_default"] = default
        elif tuned:
            all_results[name + "_tuned"] = tuned
        elif default:
            all_results[name + "_default"] = default

    # Pick best overall
    best_key = max(all_results, key=lambda k: all_results[k].mean_f1)
    best_result = all_results[best_key]
    best_model_name = best_key.rsplit("_", 1)[0]  # Remove _tuned/_default suffix

    logger.info("[%s] Model comparison:", ticker)
    for name, res in sorted(all_results.items(), key=lambda x: -x[1].mean_f1):
        logger.info("  %s: F1=%.3f±%.3f, Acc=%.3f", name, res.mean_f1,
                     np.std([r.f1_weighted for r in res.fold_results]), res.mean_accuracy)
    logger.info("[%s] BEST: %s (F1=%.3f)", ticker, best_key, best_result.mean_f1)

    # ── Train final model ───────────────────────────────────────────
    is_tuned = best_key.endswith("_tuned")
    if is_tuned:
        final_trainer = trainer
    else:
        final_trainer = default_trainer

    final_model = final_trainer.train_final_model(best_model_name, X_cv, y_cv, class_weights)

    # ── Holdout evaluation ──────────────────────────────────────────
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    preds, probs = final_trainer.predict(final_model, X_holdout)
    y_hold_int = y_holdout.astype(int)

    holdout_acc = accuracy_score(y_hold_int, preds)
    holdout_f1 = f1_score(y_hold_int, preds, average="weighted", zero_division=0)

    logger.info("[%s] Holdout: Acc=%.3f, F1=%.3f", ticker, holdout_acc, holdout_f1)
    logger.info("[%s] Holdout report:\n%s", ticker,
                classification_report(y_hold_int, preds,
                                     target_names=["SELL", "HOLD", "BUY"], zero_division=0))

    # ── SHAP analysis ───────────────────────────────────────────────
    shap_dir = OUTPUT_DIR / "shap" / safe_name
    shap_analyzer = SHAPAnalyzer(shap_dir)
    shap_result = shap_analyzer.analyze(final_model, X_holdout, f"{safe_name}_{best_model_name}")
    report = shap_analyzer.generate_report(shap_result, X_holdout)
    try:
        shap_analyzer.save_plots(shap_result, X_holdout)
    except Exception as e:
        logger.warning("[%s] SHAP plots failed: %s", ticker, e)

    # Feature stability
    fold_importances = [r.feature_importance for r in best_result.fold_results]
    validator = PatternValidator()
    stability = validator.validate_feature_stability(fold_importances)

    # ── Save model ──────────────────────────────────────────────────
    model_path = OUTPUT_DIR / f"{safe_name}_{best_model_name}.joblib"
    final_trainer.save_model(final_model, best_model_name, model_path)

    # ── Save metadata ───────────────────────────────────────────────
    metadata = {
        "ticker": ticker,
        "market": market,
        "broker": broker,
        "n_samples": len(X),
        "n_features": len(X.columns),
        "features": X.columns.tolist(),
        "label_method": label_method,
        "cost_pct": cost_pct,
        "best_model": best_key,
        "cv_f1": best_result.mean_f1,
        "cv_accuracy": best_result.mean_accuracy,
        "holdout_accuracy": holdout_acc,
        "holdout_f1": holdout_f1,
        "feature_stability": stability,
        "top_features": shap_result["top_features"][:10],
        "model_path": str(model_path),
        "feature_importance": best_result.feature_importance,
        "is_tuned": is_tuned,
    }
    if is_tuned:
        metadata["tuned_params"] = best_lgb_params if "lightgbm" in best_key else best_xgb_params

    meta_path = OUTPUT_DIR / f"{safe_name}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("[%s] DONE — model=%s, CV F1=%.3f, Holdout F1=%.3f, Stable=%s",
                ticker, best_key, best_result.mean_f1, holdout_f1, stability["stable"])

    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optuna-trials", type=int, default=30, help="Optuna trials per model")
    parser.add_argument("--markets", nargs="+", default=["india", "us", "crypto"])
    parser.add_argument("--label-method", default="fixed_threshold")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for market in args.markets:
        cfg = TICKERS[market]
        logger.info("\n" + "#" * 70)
        logger.info("MARKET: %s — %d tickers", market.upper(), len(cfg["tickers"]))
        logger.info("#" * 70)

        for ticker in cfg["tickers"]:
            df = fetch_data(ticker, market, cfg["interval"], cfg["days"])
            if df is None or df.empty:
                logger.warning("No data for %s — skipping", ticker)
                continue

            result = train_single_ticker(
                ticker=ticker,
                market=market,
                df=df,
                broker=cfg["broker"],
                optuna_trials=args.optuna_trials,
                label_method=args.label_method,
            )
            if result:
                all_results.append(result)

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TRAINING SUMMARY")
    print("=" * 80)
    print(f"\n{'Ticker':<15} {'Market':<8} {'Model':<25} {'CV F1':<10} {'Holdout F1':<12} {'Stable'}")
    print("-" * 80)

    for r in sorted(all_results, key=lambda x: -x["holdout_f1"]):
        print(f"{r['ticker']:<15} {r['market']:<8} {r['best_model']:<25} "
              f"{r['cv_f1']:<10.3f} {r['holdout_f1']:<12.3f} "
              f"{'YES' if r['feature_stability']['stable'] else 'NO'}")

    # Save summary
    summary_path = OUTPUT_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSummary saved: {summary_path}")

    # Best overall
    if all_results:
        best = max(all_results, key=lambda x: x["holdout_f1"])
        print(f"\nBEST OVERALL: {best['ticker']} ({best['market']}) — "
              f"{best['best_model']} — Holdout F1={best['holdout_f1']:.3f}")


if __name__ == "__main__":
    main()
