"""ML model training, prediction, and serialization.

Supports LightGBM, XGBoost, and RandomForest with proper regularization
for financial time-series data.
"""

import logging
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Result from a single model evaluation."""
    model_name: str
    fold_idx: int
    accuracy: float
    f1_weighted: float
    predictions: np.ndarray
    probabilities: np.ndarray | None
    feature_importance: dict[str, float] = field(default_factory=dict)


@dataclass
class EnsembleResult:
    """Aggregated result across all folds and models."""
    model_name: str
    mean_accuracy: float
    std_accuracy: float
    mean_f1: float
    fold_results: list[ModelResult] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)


class TradingModelTrainer:
    """Trains and evaluates gradient-boosted tree models for trading signals."""

    # Default hyperparameters — conservative to prevent overfitting on small datasets
    LIGHTGBM_DEFAULTS = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "max_depth": 3,
        "num_leaves": 8,
        "n_estimators": 150,
        "learning_rate": 0.01,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": 1,
    }

    XGBOOST_DEFAULTS = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "max_depth": 3,
        "n_estimators": 150,
        "learning_rate": 0.01,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbosity": 0,
        "random_state": 42,
        "n_jobs": 1,
        "tree_method": "hist",
    }

    RF_DEFAULTS = {
        "n_estimators": 200,
        "max_depth": 4,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.models_to_train = cfg.get("models", ["lightgbm", "xgboost", "random_forest"])
        self.lgb_params = {**self.LIGHTGBM_DEFAULTS, **cfg.get("lightgbm_params", {})}
        self.xgb_params = {**self.XGBOOST_DEFAULTS, **cfg.get("xgboost_params", {})}
        self.rf_params = {**self.RF_DEFAULTS, **cfg.get("rf_params", {})}

    def train_and_evaluate(
        self,
        folds: list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]],
        class_weights: dict[int, float] | None = None,
    ) -> dict[str, EnsembleResult]:
        """
        Train all configured models across all walk-forward folds.

        folds: list of (X_train, y_train, X_test, y_test) from WalkForwardCV
        Returns: dict mapping model_name -> EnsembleResult
        """
        results = {}

        for model_name in self.models_to_train:
            fold_results = []

            for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(folds):
                try:
                    result = self._train_fold(
                        model_name, fold_idx, X_train, y_train, X_test, y_test, class_weights)
                    if result:
                        fold_results.append(result)
                except Exception as e:
                    logger.warning("[%s] Fold %d failed: %s", model_name, fold_idx, e)
                    continue

            # Aggregate across folds
            accs = [r.accuracy for r in fold_results]
            f1s = [r.f1_weighted for r in fold_results]

            # Average feature importance across folds
            avg_importance = {}
            for r in fold_results:
                for feat, imp in r.feature_importance.items():
                    avg_importance[feat] = avg_importance.get(feat, 0) + imp / len(fold_results)

            results[model_name] = EnsembleResult(
                model_name=model_name,
                mean_accuracy=np.mean(accs),
                std_accuracy=np.std(accs),
                mean_f1=np.mean(f1s),
                fold_results=fold_results,
                feature_importance=dict(sorted(avg_importance.items(), key=lambda x: -x[1])),
            )

            logger.info(
                "[%s] Overall: accuracy=%.3f±%.3f, f1=%.3f",
                model_name, np.mean(accs), np.std(accs), np.mean(f1s),
            )

        return results

    def train_final_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        class_weights: dict[int, float] | None = None,
    ):
        """Train a final model on all CV data (before holdout evaluation)."""
        y_mapped = y_train.map({-1: 0, 0: 1, 1: 2}).astype(int)
        model = self._create_model(model_name, class_weights)
        model.fit(X_mapped := X_train, y_mapped)
        logger.info("Final %s model trained on %d samples", model_name, len(X_train))
        return model

    def save_model(self, model, model_name: str, path: str | Path):
        """Serialize model to disk."""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        logger.info("Model saved: %s → %s", model_name, path)

    def load_model(self, path: str | Path):
        """Load model from disk."""
        import joblib
        return joblib.load(path)

    def predict(self, model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Predict using a trained model.
        Returns (predictions, probabilities) with labels mapped back to -1, 0, 1.
        """
        preds_mapped = model.predict(X)
        # Map back: 0→-1, 1→0, 2→1
        label_map = {0: -1, 1: 0, 2: 1}
        preds = np.array([label_map[p] for p in preds_mapped])

        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)

        return preds, probs

    def _train_fold(
        self,
        model_name: str,
        fold_idx: int,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        class_weights: dict[int, float] | None = None,
    ) -> ModelResult | None:
        """Train and evaluate a single fold. Returns None if fold is skipped."""
        # Remap labels: -1,0,1 → 0,1,2
        y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2}).astype(int)
        y_test_mapped = y_test.map({-1: 0, 0: 1, 1: 2}).astype(int)

        if y_train_mapped.nunique() < 2:
            return None

        model = self._create_model(model_name, class_weights)

        # Build sample weights
        fit_kwargs = {}
        if model_name in ("lightgbm", "xgboost") and class_weights:
            weight_map = {0: class_weights.get(-1, 1.0),
                          1: class_weights.get(0, 1.0),
                          2: class_weights.get(1, 1.0)}
            fit_kwargs["sample_weight"] = np.array(
                [weight_map.get(int(v), 1.0) for v in y_train_mapped])

        # Early stopping: split train into train/val (80/20)
        # Only when train set is large enough (200+ samples) to afford splitting
        if model_name in ("lightgbm", "xgboost") and len(X_train) > 200:
            split_idx = int(len(X_train) * 0.8)
            X_t, X_v = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
            y_t, y_v = y_train_mapped.iloc[:split_idx], y_train_mapped.iloc[split_idx:]

            es_kwargs = dict(fit_kwargs)
            if "sample_weight" in es_kwargs:
                es_kwargs["sample_weight"] = es_kwargs["sample_weight"][:split_idx]

            if model_name == "lightgbm":
                import lightgbm as lgb
                es_kwargs["eval_set"] = [(X_v, y_v)]
                es_kwargs["callbacks"] = [lgb.early_stopping(30, verbose=False),
                                          lgb.log_evaluation(period=0)]
            elif model_name == "xgboost":
                es_kwargs["eval_set"] = [(X_v, y_v)]
                es_kwargs["verbose"] = False
                model.set_params(early_stopping_rounds=30)

            model.fit(X_t, y_t, **es_kwargs)
        else:
            model.fit(X_train, y_train_mapped, **fit_kwargs)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test_mapped, preds)
        f1 = f1_score(y_test_mapped, preds, average="weighted", zero_division=0)
        importance = self._get_feature_importance(model, X_train.columns.tolist())

        logger.info("[%s] Fold %d: accuracy=%.3f, f1=%.3f, train=%d, test=%d",
                     model_name, fold_idx, acc, f1, len(X_train), len(X_test))

        return ModelResult(
            model_name=model_name, fold_idx=fold_idx,
            accuracy=acc, f1_weighted=f1,
            predictions=preds, probabilities=probs,
            feature_importance=importance,
        )

    def _create_model(self, model_name: str, class_weights: dict[int, float] | None = None):
        """Instantiate a model with configured hyperparameters.

        Note: class_weight is NOT passed to models that have issues with missing classes.
        Instead, sample_weight is applied at fit time in train_and_evaluate().
        """
        if model_name == "lightgbm":
            import lightgbm as lgb
            params = self.lgb_params.copy()
            # Don't pass class_weight — use sample_weight at fit time instead
            params.pop("class_weight", None)
            return lgb.LGBMClassifier(**params)

        elif model_name == "xgboost":
            import xgboost as xgb
            params = self.xgb_params.copy()
            return xgb.XGBClassifier(**params)

        elif model_name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            params = self.rf_params.copy()
            # RF handles class_weight='balanced' well even with missing classes
            if class_weights:
                params["class_weight"] = "balanced"
            return RandomForestClassifier(**params)

        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _get_feature_importance(self, model, feature_names: list[str]) -> dict[str, float]:
        """Extract feature importance from a trained model."""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            total = importances.sum()
            if total > 0:
                importances = importances / total  # Normalize to sum to 1
            return dict(zip(feature_names, importances))
        return {}
