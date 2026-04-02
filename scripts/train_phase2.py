#!/usr/bin/env python3
"""
Phase 2 Training — Higher-frequency data + GRU ensemble.

Trains on 15-min/1h data for more samples, combines LightGBM + XGBoost + GRU.

Usage:
    python scripts/train_phase2.py --ticker RELIANCE --market india
    python scripts/train_phase2.py --ticker AAPL --market us
    python scripts/train_phase2.py --all
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
from src.finance.cost_model import estimate_round_trip_pct
from src.ml.feature_pipeline import FeaturePipeline
from src.ml.label_generator import LabelGenerator
from src.ml.walk_forward import auto_configure_cv
from src.ml.models import TradingModelTrainer
from src.ml.gru_model import GRUModelTrainer
from src.ml.ensemble import EnsembleStacker
from src.ml.hyperparameter_tuner import HyperparameterTuner
from src.ml.shap_analyzer import PatternValidator
from src.ml.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_phase2")

OUTPUT_DIR = Path("data/ml_models/phase2")
BROKER_MAP = {"india": "zerodha", "us": "alpaca", "crypto": "binance"}

from src.ml.ticker_config import INDIA_TICKERS_FLAT

# Phase 2: use higher frequency for more samples
PHASE2_CONFIG = {
    "india": {
        "tickers": INDIA_TICKERS_FLAT,  # 30 Nifty tickers
        "interval": "1h",  # yfinance limits intraday to 59 days
        "days": 59,
        "broker": "zerodha",
    },
}


def fetch_data(ticker: str, market: str, interval: str, days: int) -> pd.DataFrame | None:
    fetcher = DataFetcher()
    try:
        if market == "india":
            return fetcher.fetch_indian_stock(ticker, interval=interval, days=days)
        elif market == "us":
            return fetcher.fetch_us_stock(ticker, interval=interval, days=days)
    except Exception as e:
        logger.error("Fetch failed for %s: %s", ticker, e)
    return None


def train_phase2_ticker(ticker: str, market: str, df: pd.DataFrame,
                        interval: str, optuna_trials: int = 20) -> dict | None:
    """Full Phase 2 training: HPO + GRU + Ensemble."""
    safe = ticker.replace("/", "_")
    broker = BROKER_MAP[market]

    logger.info("=" * 70)
    logger.info("PHASE 2 TRAINING: %s (%s) — %d %s bars", ticker, market, len(df), interval)
    logger.info("=" * 70)

    # ── Features ────────────────────────────────────────────────────
    pipeline = FeaturePipeline()
    try:
        features = pipeline.transform(df)
        features = pipeline.remove_correlated(features)
    except ValueError as e:
        logger.error("Feature pipeline failed: %s", e)
        return None

    logger.info("Features: %d samples x %d features", len(features), len(features.columns))

    if len(features) < 150:
        logger.warning("Only %d samples for %s — need 150+", len(features), ticker)
        return None

    # ── Labels ──────────────────────────────────────────────────────
    cost_pct = estimate_round_trip_pct(broker)
    # For hourly data, reduce cost multiplier — hourly moves are smaller
    # but cumulative edge still needs to exceed costs over multiple trades
    cost_mult = 1.0 if interval in ("1h", "15m", "5m", "1m") else 2.0

    label_gen = LabelGenerator({
        "method": "fixed_threshold",
        "round_trip_cost_pct": cost_pct,
        "cost_multiplier": cost_mult,
        "horizon": 1,
    })
    labels = label_gen.generate(df)

    common_idx = features.index.intersection(labels.dropna().index)
    X = features.loc[common_idx]
    y = labels.loc[common_idx]

    if len(X) < 150:
        logger.warning("Only %d aligned samples — skipping", len(X))
        return None

    # ── Walk-forward CV ─────────────────────────────────────────────
    cv = auto_configure_cv(len(X), min_folds=4)
    X_cv, y_cv, X_holdout, y_holdout = cv.get_holdout_split(X, y, holdout_pct=0.15)
    folds = cv.split_dataframe(X_cv, y_cv)
    class_weights = label_gen.get_class_weights(y_cv)

    logger.info("CV: %d folds, %d cv samples, %d holdout samples",
                len(folds), len(X_cv), len(X_holdout))

    # ── Optuna HPO for tree models ──────────────────────────────────
    # Skip HPO if class imbalance is extreme (minority class < 10% of data)
    label_counts = y_cv.value_counts()
    min_class_pct = label_counts.min() / len(y_cv) * 100
    best_lgb = {}
    best_xgb = {}

    if min_class_pct >= 10 and optuna_trials > 0:
        logger.info("[%s] Optuna HPO (%d trials, LightGBM only)...", ticker, optuna_trials)
        tuner = HyperparameterTuner(n_trials=optuna_trials)
        best_lgb = tuner.tune_lightgbm(X_cv, y_cv, cv, class_weights)
        # XGBoost HPO causes segfaults on macOS — use defaults
        best_xgb = {}
    else:
        logger.info("[%s] Skipping Optuna — minority class is %.1f%%, using defaults",
                    ticker, min_class_pct)

    # ── Train tree models (tuned) ───────────────────────────────────
    logger.info("[%s] Training tuned tree models...", ticker)
    trainer = TradingModelTrainer({
        "models": ["lightgbm"],  # XGBoost causes segfault on macOS — use LightGBM only
        "lightgbm_params": best_lgb,
    })
    tree_results = trainer.train_and_evaluate(folds, class_weights)

    for name, res in tree_results.items():
        logger.info("  %s: F1=%.3f, Acc=%.3f", name, res.mean_f1, res.mean_accuracy)

    # ── Train GRU ───────────────────────────────────────────────────
    logger.info("[%s] Training GRU sequence model...", ticker)
    y_cv_mapped = y_cv.map({-1: 0, 0: 1, 1: 2}).astype(int)
    y_hold_mapped = y_holdout.map({-1: 0, 0: 1, 1: 2}).astype(int)

    gru_trainer = GRUModelTrainer({
        "seq_len": min(20, len(X_cv) // 10),
        "hidden_size": 64,
        "epochs": 80,
        "patience": 15,
        "batch_size": 32,
    })

    gru_class_weights = {0: class_weights.get(-1, 1.0),
                         1: class_weights.get(0, 1.0),
                         2: class_weights.get(1, 1.0)}

    # Split CV data 80/20 for GRU train/val
    gru_split = int(len(X_cv) * 0.8)
    try:
        gru_model, gru_history = gru_trainer.train(
            X_cv.iloc[:gru_split], y_cv_mapped.iloc[:gru_split],
            X_cv.iloc[gru_split:], y_cv_mapped.iloc[gru_split:],
            gru_class_weights,
        )
        # Evaluate GRU on holdout
        gru_preds, gru_probs = gru_trainer.predict(gru_model, X_holdout)
        from sklearn.metrics import accuracy_score, f1_score
        gru_acc = accuracy_score(y_hold_mapped, gru_preds[-len(y_hold_mapped):])
        gru_f1 = f1_score(y_hold_mapped, gru_preds[-len(y_hold_mapped):],
                          average="weighted", zero_division=0)
        logger.info("  GRU holdout: Acc=%.3f, F1=%.3f", gru_acc, gru_f1)
    except Exception as e:
        logger.warning("GRU training failed: %s", e)
        gru_model = None
        gru_acc = gru_f1 = 0

    # ── Ensemble ────────────────────────────────────────────────────
    logger.info("[%s] Building stacking ensemble...", ticker)
    ensemble = EnsembleStacker()
    try:
        ens_metrics = ensemble.train_stacked(folds, X_cv, y_cv, class_weights)
        logger.info("  Ensemble meta-learner: Acc=%.3f, F1=%.3f",
                     ens_metrics.get("meta_accuracy", 0), ens_metrics.get("meta_f1", 0))
    except Exception as e:
        logger.warning("Ensemble failed: %s — falling back to best tree model", e)
        ens_metrics = {"meta_f1": 0}

    # ── Holdout evaluation ──────────────────────────────────────────
    logger.info("[%s] Holdout evaluation...", ticker)

    # Best tree model
    best_tree_name = max(tree_results, key=lambda k: tree_results[k].mean_f1)
    best_tree = trainer.train_final_model(best_tree_name, X_cv, y_cv, class_weights)
    tree_preds_raw = best_tree.predict(X_holdout)
    tree_preds = np.array([{0: -1, 1: 0, 2: 1}[p] for p in tree_preds_raw])

    # Ensemble
    try:
        ens_preds, ens_probs = ensemble.predict(X_holdout)
    except Exception:
        ens_preds = tree_preds
        ens_probs = None

    from sklearn.metrics import classification_report
    y_hold_int = y_holdout.astype(int)

    tree_f1 = f1_score(y_hold_int, tree_preds, average="weighted", zero_division=0)
    ens_f1 = f1_score(y_hold_int, ens_preds, average="weighted", zero_division=0)

    logger.info("  Tree holdout F1: %.3f", tree_f1)
    logger.info("  Ensemble holdout F1: %.3f", ens_f1)
    logger.info("  GRU holdout F1: %.3f", gru_f1)

    # Pick best approach
    best_f1 = max(tree_f1, ens_f1, gru_f1)
    if ens_f1 >= tree_f1 and ens_f1 >= gru_f1:
        best_approach = "ensemble"
        best_preds = ens_preds
    elif gru_f1 > tree_f1:
        best_approach = "gru"
        best_preds = gru_preds[-len(y_hold_int):]
    else:
        best_approach = f"tree_{best_tree_name}"
        best_preds = tree_preds

    logger.info("  BEST: %s (F1=%.3f)", best_approach, best_f1)

    print(f"\n[{ticker}] Holdout Report ({best_approach}):")
    print(classification_report(y_hold_int, best_preds,
                                target_names=["SELL", "HOLD", "BUY"], zero_division=0))

    # ── SHAP (skip if crashing — can run separately) ──────────────
    # Feature importance from tree model (skip SHAP — causes segfault on macOS)
    fi = tree_results[best_tree_name].feature_importance
    top_features_list = list(fi.items())[:10]
    fold_imps = [r.feature_importance for r in tree_results[best_tree_name].fold_results]
    stability = PatternValidator().validate_feature_stability(fold_imps)

    # ── Save ────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save ensemble
    ensemble.save(OUTPUT_DIR / f"{safe}_ensemble")

    # Save tree model
    import joblib
    tree_path = OUTPUT_DIR / f"{safe}_{best_tree_name}.joblib"
    joblib.dump(best_tree, tree_path)

    # Save GRU
    if gru_model:
        gru_trainer.save(gru_model, OUTPUT_DIR / f"{safe}_gru.pt")

    # Metadata
    metadata = {
        "ticker": ticker,
        "market": market,
        "phase": 2,
        "interval": interval,
        "n_samples": len(X),
        "n_features": len(X.columns),
        "features": X.columns.tolist(),
        "cost_pct": cost_pct,
        "best_approach": best_approach,
        "tree_model": best_tree_name,
        "tree_f1": tree_f1,
        "gru_f1": gru_f1,
        "ensemble_f1": ens_f1,
        "best_f1": best_f1,
        "feature_stability": stability,
        "top_features": top_features_list,
        "optuna_lgb_params": best_lgb,
        "optuna_xgb_params": best_xgb,
    }
    (OUTPUT_DIR / f"{safe}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    logger.info("[%s] PHASE 2 COMPLETE: best=%s, F1=%.3f, samples=%d (%s)",
                ticker, best_approach, best_f1, len(X), interval)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", help="Single ticker")
    parser.add_argument("--market", choices=["india", "us"])
    parser.add_argument("--all", action="store_true", help="Train all configured tickers")
    parser.add_argument("--optuna-trials", type=int, default=20)
    args = parser.parse_args()

    results = []

    if args.all or (not args.ticker):
        for market, cfg in PHASE2_CONFIG.items():
            if args.market and market != args.market:
                continue
            for ticker in cfg["tickers"]:
                df = fetch_data(ticker, market, cfg["interval"], cfg["days"])
                if df is None or df.empty:
                    logger.warning("No data for %s", ticker)
                    continue
                r = train_phase2_ticker(ticker, market, df, cfg["interval"], args.optuna_trials)
                if r:
                    results.append(r)
    else:
        market = args.market or "us"
        cfg = PHASE2_CONFIG[market]
        df = fetch_data(args.ticker, market, cfg["interval"], cfg["days"])
        if df is not None:
            r = train_phase2_ticker(args.ticker, market, df, cfg["interval"], args.optuna_trials)
            if r:
                results.append(r)

    # Summary
    if results:
        print("\n" + "=" * 80)
        print("PHASE 2 TRAINING SUMMARY")
        print("=" * 80)
        print(f"\n{'Ticker':<12} {'Market':<8} {'Interval':<10} {'Samples':<10} "
              f"{'Best':<15} {'Tree F1':<10} {'GRU F1':<10} {'Ens F1':<10} {'Best F1':<10}")
        print("-" * 100)
        for r in sorted(results, key=lambda x: -x["best_f1"]):
            print(f"{r['ticker']:<12} {r['market']:<8} {r['interval']:<10} {r['n_samples']:<10} "
                  f"{r['best_approach']:<15} {r['tree_f1']:<10.3f} {r['gru_f1']:<10.3f} "
                  f"{r.get('ensemble_f1', 0):<10.3f} {r['best_f1']:<10.3f}")

        (OUTPUT_DIR / "phase2_summary.json").write_text(json.dumps(results, indent=2, default=str))
        print(f"\nSummary saved: {OUTPUT_DIR / 'phase2_summary.json'}")


if __name__ == "__main__":
    main()
