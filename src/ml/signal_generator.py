"""ML-based signal generator — replaces rule-based signals.

Loads a trained model, runs the feature pipeline on live/historical data,
and generates BUY/HOLD/SELL signals with confidence scores and SHAP explanations.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..finance.base_trader import Signal, SignalAction
from .feature_pipeline import FeaturePipeline
from .models import TradingModelTrainer

logger = logging.getLogger(__name__)


class MLSignalGenerator:
    """Generates trading signals using a trained ML model."""

    def __init__(self, model_path: str | Path, config: dict | None = None):
        cfg = config or {}
        self.model_path = Path(model_path)
        self.confidence_threshold = cfg.get("confidence_threshold", 0.40)
        self.feature_pipeline = FeaturePipeline(cfg.get("features", {}))

        # Load trained model
        trainer = TradingModelTrainer()
        self.model = trainer.load_model(self.model_path)
        logger.info("Loaded ML model from %s", self.model_path)

        # Load metadata if available
        meta_path = self.model_path.with_suffix(".joblib").parent / (
            self.model_path.stem.rsplit("_", 1)[0] + "_metadata.json"
        )
        self.metadata = {}
        if meta_path.exists():
            import json
            self.metadata = json.loads(meta_path.read_text())
            logger.info("Loaded metadata: %d features, cost=%.4f%%",
                        self.metadata.get("n_features", 0),
                        self.metadata.get("cost_pct", 0) * 100)

    def generate(self, df: pd.DataFrame, ticker: str) -> Signal:
        """
        Generate a trading signal from OHLCV data.

        df: DataFrame with at least 60 bars of OHLCV data
        ticker: symbol for the signal
        """
        if df is None or len(df) < 60:
            return self._hold_signal(ticker, "Insufficient data")

        try:
            # Run feature pipeline
            features = self.feature_pipeline.transform(df)
            features = features.replace([np.inf, -np.inf], np.nan).dropna()

            if len(features) == 0:
                return self._hold_signal(ticker, "No valid features after transform")

            # Filter to EXACT features the model was trained on (order matters)
            trained_features = self.metadata.get("features", [])
            if trained_features:
                missing = set(trained_features) - set(features.columns)
                if missing:
                    logger.error("Missing %d features for inference: %s", len(missing), missing)
                    return self._hold_signal(ticker, f"Missing {len(missing)} features")
                features = features[trained_features]  # Enforce exact order

            # Use only the latest bar's features for prediction
            X_latest = features.iloc[[-1]]

            # Predict
            preds_mapped = self.model.predict(X_latest)
            label_map = {0: -1, 1: 0, 2: 1}
            prediction = label_map[int(preds_mapped[0])]

            # Get probability (confidence)
            confidence = 0.0
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X_latest)[0]
                confidence = float(probs.max()) * 100  # Convert to 0-100 scale

            # Determine action
            if prediction == 1 and confidence >= self.confidence_threshold:
                action = SignalAction.BUY
            elif prediction == -1 and confidence >= self.confidence_threshold:
                action = SignalAction.SELL
            else:
                action = SignalAction.HOLD

            # Get price and compute stop-loss/take-profit from ATR
            price = float(df["Close"].iloc[-1])
            import pandas_ta as ta
            atr = ta.atr(df["High"], df["Low"], df["Close"], length=14)
            atr_val = float(atr.iloc[-1]) if atr is not None and not pd.isna(atr.iloc[-1]) else price * 0.02

            if action == SignalAction.BUY:
                stop_loss = round(price - 2 * atr_val, 2)
                take_profit = round(price + 3 * atr_val, 2)
            elif action == SignalAction.SELL:
                stop_loss = round(price + 2 * atr_val, 2)
                take_profit = round(price - 3 * atr_val, 2)
            else:
                stop_loss = None
                take_profit = None

            # Build reasoning from top features
            reasoning = self._build_reasoning(X_latest, prediction, confidence)

            signal = Signal(
                ticker=ticker,
                action=action,
                confidence=round(confidence, 1),
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=reasoning,
                technical_score=float(prediction) * confidence,
                sentiment_score=0.0,  # ML model doesn't use sentiment
                chart_score=0.0,
            )

            logger.info(
                "ML Signal %s: %s (conf=%.1f%%) pred=%d | %s",
                ticker, action.value, confidence, prediction, reasoning[:100],
            )
            return signal

        except Exception as e:
            logger.error("ML signal generation failed for %s: %s", ticker, e)
            return self._hold_signal(ticker, f"Error: {e}")

    def generate_batch(self, data: dict[str, pd.DataFrame]) -> list[Signal]:
        """Generate signals for multiple tickers."""
        signals = []
        for ticker, df in data.items():
            signal = self.generate(df, ticker)
            if signal.action != SignalAction.HOLD:
                signals.append(signal)
        return signals

    def _build_reasoning(self, X: pd.DataFrame, prediction: int, confidence: float) -> str:
        """Build human-readable reasoning from feature values."""
        parts = []
        direction = "BUY" if prediction == 1 else "SELL" if prediction == -1 else "HOLD"
        parts.append(f"ML:{direction}(conf={confidence:.0f}%)")

        # Show top feature values
        if hasattr(self.model, "feature_importances_"):
            importances = dict(zip(X.columns, self.model.feature_importances_))
            top = sorted(importances.items(), key=lambda x: -x[1])[:5]
            feat_strs = []
            for feat, imp in top:
                val = X[feat].iloc[0]
                feat_strs.append(f"{feat}={val:.3f}")
            parts.append("Top: " + ", ".join(feat_strs))

        return " | ".join(parts)

    def _hold_signal(self, ticker: str, reason: str) -> Signal:
        return Signal(
            ticker=ticker,
            action=SignalAction.HOLD,
            confidence=0,
            price=0,
            reasoning=f"ML:HOLD ({reason})",
        )
