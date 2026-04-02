"""Ensemble stacking — combines LightGBM + XGBoost + GRU predictions.

Uses a Ridge Regression meta-learner (simple, low overfitting risk)
trained on out-of-fold predictions from base models.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score, accuracy_score
import joblib

logger = logging.getLogger(__name__)


class EnsembleStacker:
    """Stacking ensemble combining tree-based and sequence model predictions."""

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.meta_learner = None
        self.base_models = {}
        self.gru_trainer = None
        self.gru_model = None
        self.weights = cfg.get("weights", {
            "lightgbm": 0.40,
            "xgboost": 0.30,
            "gru": 0.30,
        })

    def train_stacked(
        self,
        folds: list[tuple],
        X_full: pd.DataFrame,
        y_full: pd.Series,
        class_weights: dict | None = None,
    ) -> dict:
        """
        Train base models + meta-learner using out-of-fold predictions.

        folds: list of (X_train, y_train, X_test, y_test) from walk-forward CV
        Returns: performance metrics
        """
        from .models import TradingModelTrainer
        from .gru_model import GRUModelTrainer

        n_samples = len(X_full)
        n_classes = 3

        # Collect out-of-fold predictions for each base model
        oof_preds = {
            "lightgbm": np.zeros((n_samples, n_classes)),
            "xgboost": np.zeros((n_samples, n_classes)),
        }
        oof_mask = np.zeros(n_samples, dtype=bool)

        # Train tree models across folds (LightGBM only — XGBoost segfaults on macOS)
        tree_models = ["lightgbm"]
        trainer = TradingModelTrainer({"models": tree_models})

        for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(folds):
            y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2}).astype(int)
            y_test_mapped = y_test.map({-1: 0, 0: 1, 1: 2}).astype(int)

            for model_name in tree_models:
                try:
                    result = trainer._train_fold(
                        model_name, fold_idx, X_train, y_train, X_test, y_test, class_weights)
                    if result and result.probabilities is not None:
                        # Map test indices back to full array
                        test_indices = X_full.index.get_indexer(X_test.index)
                        valid = test_indices >= 0
                        probs = result.probabilities
                        if probs.shape[1] == n_classes:
                            oof_preds[model_name][test_indices[valid]] = probs[valid]
                            oof_mask[test_indices[valid]] = True
                except Exception as e:
                    logger.warning("Fold %d %s failed: %s", fold_idx, model_name, e)

        # Train GRU on full CV data with early stopping
        gru_trainer = GRUModelTrainer({
            "seq_len": min(20, len(X_full) // 10),
            "hidden_size": 64,
            "epochs": 80,
            "patience": 15,
        })

        y_mapped = y_full.map({-1: 0, 0: 1, 1: 2}).astype(int)

        # Split for GRU validation
        split = int(len(X_full) * 0.8)
        X_gru_train, X_gru_val = X_full.iloc[:split], X_full.iloc[split:]
        y_gru_train, y_gru_val = y_mapped.iloc[:split], y_mapped.iloc[split:]

        gru_class_weights = {0: class_weights.get(-1, 1.0),
                             1: class_weights.get(0, 1.0),
                             2: class_weights.get(1, 1.0)} if class_weights else None

        try:
            gru_model, gru_history = gru_trainer.train(
                X_gru_train, y_gru_train, X_gru_val, y_gru_val, gru_class_weights)
            self.gru_model = gru_model
            self.gru_trainer = gru_trainer

            # Get GRU predictions on validation portion
            gru_preds, gru_probs = gru_trainer.predict(gru_model, X_full)
            oof_preds["gru"] = np.zeros((n_samples, n_classes))
            # GRU predictions are offset by seq_len
            offset = n_samples - len(gru_probs)
            if len(gru_probs) > 0 and gru_probs.shape[1] == n_classes:
                oof_preds["gru"][offset:] = gru_probs

            logger.info("GRU trained: val_acc=%.3f",
                        gru_history.get("val_acc", [0])[-1] if gru_history.get("val_acc") else 0)
        except Exception as e:
            logger.warning("GRU training failed: %s — using tree-only ensemble", e)
            oof_preds["gru"] = np.zeros((n_samples, n_classes))

        # Build meta-features from OOF predictions
        meta_features = np.hstack([oof_preds[name] for name in sorted(oof_preds.keys())])
        meta_X = meta_features[oof_mask]
        meta_y = y_mapped.values[oof_mask]

        if len(meta_X) < 10:
            logger.warning("Not enough OOF predictions for meta-learner (%d)", len(meta_X))
            return {"method": "weighted_average", "accuracy": 0}

        # Train meta-learner (Ridge — simple, low overfitting)
        self.meta_learner = RidgeClassifier(alpha=1.0)
        self.meta_learner.fit(meta_X, meta_y)

        meta_preds = self.meta_learner.predict(meta_X)
        meta_acc = accuracy_score(meta_y, meta_preds)
        meta_f1 = f1_score(meta_y, meta_preds, average="weighted", zero_division=0)

        logger.info("Meta-learner trained: acc=%.3f, f1=%.3f on %d OOF samples",
                     meta_acc, meta_f1, len(meta_X))

        # Also train final base models on full data for inference
        self._train_final_base_models(X_full, y_full, class_weights)

        return {
            "method": "stacking",
            "meta_accuracy": meta_acc,
            "meta_f1": meta_f1,
            "oof_samples": len(meta_X),
        }

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict using the ensemble.

        Returns: (predictions as -1/0/1, probabilities shape (n, 3))
        """
        n_classes = 3
        all_probs = {}

        # Tree model predictions
        for name, model in self.base_models.items():
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                if probs.shape[1] == n_classes:
                    all_probs[name] = probs

        # GRU predictions
        if self.gru_model and self.gru_trainer:
            try:
                _, gru_probs = self.gru_trainer.predict(self.gru_model, X)
                if len(gru_probs) > 0 and gru_probs.shape[1] == n_classes:
                    # Align GRU output (may be shorter due to seq_len)
                    full_probs = np.ones((len(X), n_classes)) / n_classes
                    offset = len(X) - len(gru_probs)
                    full_probs[offset:] = gru_probs
                    all_probs["gru"] = full_probs
            except Exception as e:
                logger.warning("GRU prediction failed: %s", e)

        if not all_probs:
            return np.zeros(len(X)), np.ones((len(X), n_classes)) / n_classes

        # Use meta-learner if available
        if self.meta_learner and len(all_probs) >= 2:
            meta_features = np.hstack([all_probs[name] for name in sorted(all_probs.keys())])
            preds_mapped = self.meta_learner.predict(meta_features)
        else:
            # Fallback: weighted average
            weighted_probs = np.zeros((len(X), n_classes))
            total_weight = 0
            for name, probs in all_probs.items():
                w = self.weights.get(name, 1.0 / len(all_probs))
                weighted_probs += probs * w
                total_weight += w
            weighted_probs /= total_weight
            preds_mapped = np.argmax(weighted_probs, axis=1)

        # Map back: 0→-1, 1→0, 2→1
        label_map = {0: -1, 1: 0, 2: 1}
        preds = np.array([label_map[int(p)] for p in preds_mapped])

        # Average probabilities for confidence
        avg_probs = np.mean(list(all_probs.values()), axis=0)

        return preds, avg_probs

    def _train_final_base_models(self, X: pd.DataFrame, y: pd.Series,
                                  class_weights: dict | None):
        """Train final tree models on full data for inference."""
        from .models import TradingModelTrainer

        trainer = TradingModelTrainer({"models": ["lightgbm"]})
        for name in ["lightgbm"]:
            model = trainer.train_final_model(name, X, y, class_weights)
            self.base_models[name] = model

    def save(self, path: str | Path):
        """Save the full ensemble."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save tree models
        for name, model in self.base_models.items():
            joblib.dump(model, path / f"base_{name}.joblib")

        # Save meta-learner
        if self.meta_learner:
            joblib.dump(self.meta_learner, path / "meta_learner.joblib")

        # Save GRU
        if self.gru_model and self.gru_trainer:
            self.gru_trainer.save(self.gru_model, path / "gru_model.pt")

        # Save config
        import json
        config = {"weights": self.weights, "base_models": list(self.base_models.keys())}
        (path / "ensemble_config.json").write_text(json.dumps(config, indent=2))

        logger.info("Ensemble saved: %s", path)

    def load(self, path: str | Path):
        """Load a saved ensemble."""
        import json
        from .gru_model import GRUModelTrainer

        path = Path(path)
        config = json.loads((path / "ensemble_config.json").read_text())
        self.weights = config["weights"]

        # Load tree models
        for name in config["base_models"]:
            model_path = path / f"base_{name}.joblib"
            if model_path.exists():
                self.base_models[name] = joblib.load(model_path)

        # Load meta-learner
        meta_path = path / "meta_learner.joblib"
        if meta_path.exists():
            self.meta_learner = joblib.load(meta_path)

        # Load GRU
        gru_path = path / "gru_model.pt"
        if gru_path.exists():
            self.gru_trainer = GRUModelTrainer()
            self.gru_model = self.gru_trainer.load(gru_path)

        logger.info("Ensemble loaded from %s", path)
