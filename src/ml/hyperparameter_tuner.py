"""Optuna-based hyperparameter optimization for trading models.

Tunes model hyperparameters using walk-forward CV to avoid overfitting.
"""

import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score

from .walk_forward import WalkForwardCV, auto_configure_cv

logger = logging.getLogger(__name__)

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterTuner:
    """Tune ML model hyperparameters with Optuna + walk-forward CV."""

    def __init__(self, n_trials: int = 50, n_jobs: int = 1):
        self.n_trials = n_trials
        self.n_jobs = n_jobs

    def tune_lightgbm(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: WalkForwardCV,
        class_weights: dict | None = None,
    ) -> dict:
        """Tune LightGBM hyperparameters."""
        import lightgbm as lgb

        folds = cv.split_dataframe(X, y)

        def objective(trial):
            params = {
                "objective": "multiclass",
                "num_class": 3,
                "metric": "multi_logloss",
                "boosting_type": "gbdt",
                "max_depth": trial.suggest_int("max_depth", 2, 5),
                "num_leaves": trial.suggest_int("num_leaves", 4, 16),
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 40),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 2.0, log=True),
                "verbose": -1,
                "random_state": 42,
                "n_jobs": 1,  # Prevent memory issues
                "force_col_wise": True,
            }
            # Use sample_weight instead of class_weight to avoid class mapping issues
            weight_map = None
            if class_weights:
                weight_map = {0: class_weights.get(-1, 1.0),
                              1: class_weights.get(0, 1.0),
                              2: class_weights.get(1, 1.0)}

            scores = []
            for X_train, y_train, X_test, y_test in folds:
                y_tr = y_train.map({-1: 0, 0: 1, 1: 2}).astype(int)
                y_te = y_test.map({-1: 0, 0: 1, 1: 2}).astype(int)
                model = lgb.LGBMClassifier(**params)
                sw = np.array([weight_map.get(int(v), 1.0) for v in y_tr]) if weight_map else None
                model.fit(X_train, y_tr, sample_weight=sw)
                preds = model.predict(X_test)
                scores.append(f1_score(y_te, preds, average="weighted", zero_division=0))

            return np.mean(scores)

        study = optuna.create_study(direction="maximize", study_name="lightgbm_tune")
        try:
            study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs,
                           timeout=120)  # 2-minute timeout
        except Exception as e:
            logger.warning("LightGBM HPO failed: %s — using defaults", e)
            return {}

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed:
            logger.info("LightGBM best F1=%.4f (%d/%d trials ok)", study.best_value,
                        len(completed), len(study.trials))
            return study.best_params
        logger.warning("LightGBM HPO: no successful trials — using defaults")
        return {}

    def tune_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: WalkForwardCV,
        class_weights: dict | None = None,
    ) -> dict:
        """Tune XGBoost hyperparameters."""
        import xgboost as xgb

        folds = cv.split_dataframe(X, y)

        def objective(trial):
            params = {
                "objective": "multi:softprob",
                "num_class": 3,
                "eval_metric": "mlogloss",
                "max_depth": trial.suggest_int("max_depth", 2, 5),
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                "min_child_weight": trial.suggest_int("min_child_weight", 10, 40),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 2.0, log=True),
                "verbosity": 0,
                "random_state": 42,
                "n_jobs": 1,
                "tree_method": "hist",
            }

            weight_map = None
            if class_weights:
                weight_map = {0: class_weights.get(-1, 1.0),
                              1: class_weights.get(0, 1.0),
                              2: class_weights.get(1, 1.0)}

            scores = []
            for X_train, y_train, X_test, y_test in folds:
                y_tr = y_train.map({-1: 0, 0: 1, 1: 2}).astype(int)
                y_te = y_test.map({-1: 0, 0: 1, 1: 2}).astype(int)

                # Skip fold if fewer than 2 classes in train
                if y_tr.nunique() < 2:
                    continue

                # Remap to contiguous classes if some are missing
                unique_classes = sorted(y_tr.unique())
                if unique_classes != [0, 1, 2]:
                    remap = {c: i for i, c in enumerate(unique_classes)}
                    inv_remap = {i: c for c, i in remap.items()}
                    y_tr_fit = y_tr.map(remap)
                    y_te_fit = y_te.map(lambda x: remap.get(x, -1))
                    valid_test = y_te_fit >= 0
                    y_te_fit = y_te_fit[valid_test]
                    X_test_fit = X_test[valid_test]
                    n_class = len(unique_classes)
                else:
                    y_tr_fit = y_tr
                    y_te_fit = y_te
                    X_test_fit = X_test
                    n_class = 3

                p = params.copy()
                p["num_class"] = n_class
                if n_class == 2:
                    p["objective"] = "binary:logistic"
                    del p["num_class"]

                model = xgb.XGBClassifier(**p)
                sw = np.array([weight_map.get(int(v), 1.0) for v in y_tr]) if weight_map else None
                try:
                    model.fit(X_train, y_tr_fit, sample_weight=sw)
                    preds = model.predict(X_test_fit)
                    scores.append(f1_score(y_te_fit, preds, average="weighted", zero_division=0))
                except Exception:
                    continue

            return np.mean(scores) if scores else 0.0

        study = optuna.create_study(direction="maximize", study_name="xgboost_tune")
        try:
            study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs,
                           timeout=120)
        except Exception as e:
            logger.warning("XGBoost HPO failed: %s — using defaults", e)
            return {}

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed:
            logger.info("XGBoost best F1=%.4f (%d/%d trials ok)", study.best_value,
                        len(completed), len(study.trials))
            return study.best_params
        logger.warning("XGBoost HPO: no successful trials — using defaults")
        return {}
