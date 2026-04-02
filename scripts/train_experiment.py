#!/usr/bin/env python3
"""Train models for all experiment timeframes."""

import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.finance.analysis.data_fetcher import DataFetcher
from src.finance.cost_model import estimate_round_trip_pct
from src.ml.feature_pipeline import FeaturePipeline
from src.ml.label_generator import LabelGenerator
from src.ml.walk_forward import auto_configure_cv
from src.ml.models import TradingModelTrainer
from src.ml.hyperparameter_tuner import HyperparameterTuner
from scripts.experiment_config import TIMEFRAME_CONFIG, EXPERIMENT_TICKERS, get_arm_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_experiment")


def train_ticker_for_timeframe(ticker: str, tf: str, cfg: dict, model_dir: Path) -> dict | None:
    """Train a single ticker for a specific timeframe."""
    fetcher = DataFetcher()
    df = fetcher.fetch_indian_stock(ticker, cfg["interval"], cfg["days"])
    if df is None or len(df) < 60:
        return None

    pipeline = FeaturePipeline()
    try:
        features = pipeline.transform(df)
        features = pipeline.remove_correlated(features)
    except ValueError:
        return None

    cost = estimate_round_trip_pct("zerodha")
    labels = LabelGenerator({
        "round_trip_cost_pct": cost,
        "cost_multiplier": cfg["cost_mult"],
    }).generate(df)

    common = features.index.intersection(labels.dropna().index)
    X, y = features.loc[common], labels.loc[common]

    if len(X) < cfg["min_samples"]:
        logger.warning("[%s][%s] Only %d samples — skipping", tf, ticker, len(X))
        return None

    cv = auto_configure_cv(len(X), min_folds=3)
    X_cv, y_cv, X_hold, y_hold = cv.get_holdout_split(X, y, 0.15)
    folds = cv.split_dataframe(X_cv, y_cv)
    cw = LabelGenerator({"round_trip_cost_pct": cost}).get_class_weights(y_cv)

    # Optuna HPO (LightGBM only)
    best_params = {}
    label_counts = y_cv.value_counts()
    min_pct = label_counts.min() / len(y_cv) * 100
    if min_pct >= 10 and cfg["optuna_trials"] > 0:
        tuner = HyperparameterTuner(n_trials=cfg["optuna_trials"])
        best_params = tuner.tune_lightgbm(X_cv, y_cv, cv, cw)

    trainer = TradingModelTrainer({"models": ["lightgbm"], "lightgbm_params": best_params})
    results = trainer.train_and_evaluate(folds, cw)

    if not results or "lightgbm" not in results:
        return None

    cv_f1 = results["lightgbm"].mean_f1

    # Final model + holdout
    final = trainer.train_final_model("lightgbm", X_cv, y_cv, cw)
    from sklearn.metrics import f1_score
    preds = np.array([{0: -1, 1: 0, 2: 1}[p] for p in final.predict(X_hold)])
    hold_f1 = f1_score(y_hold.astype(int), preds, average="weighted", zero_division=0)

    # Save
    model_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(final, model_dir / f"{ticker}_lightgbm.joblib")

    meta = {
        "ticker": ticker, "timeframe": tf, "interval": cfg["interval"],
        "n_samples": len(X), "n_features": len(X.columns),
        "features": X.columns.tolist(), "cv_f1": cv_f1, "holdout_f1": hold_f1,
    }
    (model_dir / f"{ticker}_metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    logger.info("[%s][%s] F1: cv=%.3f holdout=%.3f (%d samples)", tf, ticker, cv_f1, hold_f1, len(X))
    return meta


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframes", nargs="+", default=list(TIMEFRAME_CONFIG.keys()))
    parser.add_argument("--tickers", nargs="+", default=EXPERIMENT_TICKERS)
    args = parser.parse_args()

    all_results = {}

    for tf in args.timeframes:
        if tf == "multi":
            # Multi uses models from 1d, 1h, 15m — train those if not already done
            logger.info("[multi] Uses models from 1d, 1h, 15m arms — no separate training needed")
            continue

        cfg = TIMEFRAME_CONFIG[tf]
        dirs = get_arm_dirs(tf)
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING ARM: %s (%s, %d days)", tf, cfg["interval"], cfg["days"])
        logger.info("=" * 60)

        results = []
        for ticker in args.tickers[:cfg["max_tickers"]]:
            r = train_ticker_for_timeframe(ticker, tf, cfg, dirs["models"])
            if r:
                results.append(r)

        all_results[tf] = results
        logger.info("[%s] Trained %d/%d tickers", tf, len(results), len(args.tickers))

    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT TRAINING SUMMARY")
    print("=" * 80)
    for tf, results in all_results.items():
        if not results:
            print(f"  {tf}: No models trained")
            continue
        f1s = [r["holdout_f1"] for r in results]
        print(f"  {tf}: {len(results)} models, avg F1={np.mean(f1s):.3f}, "
              f"best={max(results, key=lambda x: x['holdout_f1'])['ticker']} ({max(f1s):.3f})")


if __name__ == "__main__":
    main()
