"""Multi-timeframe hierarchical signal generator.

Combines 3 timeframes for ARM 5 of the experiment:
  1d → regime (BULLISH / BEARISH / NEUTRAL)
  1h → direction confirmation
  15m → entry timing

Only generates BUY/SELL when all timeframes agree.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from ..finance.analysis.data_fetcher import DataFetcher
from ..finance.base_trader import Signal, SignalAction
from .feature_pipeline import FeaturePipeline

logger = logging.getLogger(__name__)


class MultiTimeframeSignalGenerator:
    """Hierarchical signal: 1d regime → 1h direction → 15m entry."""

    def __init__(self, model_dirs: dict[str, Path], fetcher: DataFetcher | None = None):
        """
        model_dirs: {"1d": Path, "1h": Path, "15m": Path}
        """
        self.fetcher = fetcher or DataFetcher()
        self.pipeline = FeaturePipeline()
        self.model_dirs = model_dirs
        self.models: dict[str, dict] = {}  # tf -> {ticker -> (model, meta)}

        # Caches
        self._regime_cache: dict[str, tuple[str, float, datetime]] = {}  # ticker -> (regime, conf, time)
        self._hourly_cache: dict[str, tuple[int, float, datetime]] = {}  # ticker -> (pred, conf, time)

        self._load_models()

    def _load_models(self):
        """Load models for all 3 timeframes."""
        for tf, model_dir in self.model_dirs.items():
            self.models[tf] = {}
            if not model_dir.exists():
                logger.warning("Model dir not found for %s: %s", tf, model_dir)
                continue
            for jf in model_dir.glob("*_lightgbm.joblib"):
                ticker = jf.stem.replace("_lightgbm", "")
                meta_path = model_dir / f"{ticker}_metadata.json"
                meta = {}
                if meta_path.exists():
                    import json
                    meta = json.loads(meta_path.read_text())
                self.models[tf][ticker] = (joblib.load(jf), meta)
            logger.info("Loaded %d %s models", len(self.models[tf]), tf)

    def generate(self, ticker: str) -> Signal:
        """Generate hierarchical signal for a ticker."""
        price = 0.0

        # Step 1: Daily regime (cached, refreshed once per day)
        regime, regime_conf = self._get_regime(ticker)

        # Step 2: Hourly direction (cached, refreshed every hour)
        hourly_pred, hourly_conf = self._get_hourly(ticker)

        # Step 3: 15m entry (always fresh)
        entry_pred, entry_conf, price = self._get_entry(ticker)

        # Step 4: Hierarchical filter
        action, final_conf = self._combine(regime, regime_conf,
                                            hourly_pred, hourly_conf,
                                            entry_pred, entry_conf)

        # ATR for stop/take-profit
        import pandas_ta as ta
        df_15m = self.fetcher.fetch_indian_stock(ticker, "15m", 10)
        sl = tp = None
        if df_15m is not None and len(df_15m) > 14:
            atr = ta.atr(df_15m["High"], df_15m["Low"], df_15m["Close"], length=14)
            atr_val = float(atr.iloc[-1]) if atr is not None and not pd.isna(atr.iloc[-1]) else price * 0.01
            if action == SignalAction.BUY:
                sl = round(price - 2 * atr_val, 2)
                tp = round(price + 3 * atr_val, 2)
            elif action == SignalAction.SELL:
                sl = round(price + 2 * atr_val, 2)
                tp = round(price - 3 * atr_val, 2)

        reasoning = (f"MULTI: regime={regime}({regime_conf:.0f}%) "
                     f"1h={'BUY' if hourly_pred==1 else 'SELL' if hourly_pred==-1 else 'HOLD'}"
                     f"({hourly_conf:.0f}%) "
                     f"15m={'BUY' if entry_pred==1 else 'SELL' if entry_pred==-1 else 'HOLD'}"
                     f"({entry_conf:.0f}%)")

        return Signal(
            ticker=ticker, action=action, confidence=round(final_conf, 1),
            price=price, stop_loss=sl, take_profit=tp, reasoning=reasoning,
        )

    def _get_regime(self, ticker: str) -> tuple[str, float]:
        """Get daily regime — cached, refreshed once per day."""
        cached = self._regime_cache.get(ticker)
        if cached:
            regime, conf, cached_time = cached
            if datetime.now() - cached_time < timedelta(hours=12):
                return regime, conf

        if "1d" not in self.models or ticker not in self.models["1d"]:
            return "NEUTRAL", 50.0

        model, meta = self.models["1d"][ticker]
        df = self.fetcher.fetch_indian_stock(ticker, "1d", 365)
        if df is None or len(df) < 60:
            return "NEUTRAL", 50.0

        pred, conf = self._predict(model, meta, df)

        if pred == 1 and conf >= 35:
            regime = "BULLISH"
        elif pred == -1 and conf >= 35:
            regime = "BEARISH"
        else:
            regime = "NEUTRAL"

        self._regime_cache[ticker] = (regime, conf, datetime.now())
        return regime, conf

    def _get_hourly(self, ticker: str) -> tuple[int, float]:
        """Get hourly direction — cached, refreshed every hour."""
        cached = self._hourly_cache.get(ticker)
        if cached:
            pred, conf, cached_time = cached
            if datetime.now() - cached_time < timedelta(minutes=55):
                return pred, conf

        if "1h" not in self.models or ticker not in self.models["1h"]:
            return 0, 50.0

        model, meta = self.models["1h"][ticker]
        df = self.fetcher.fetch_indian_stock(ticker, "1h", 59)
        if df is None or len(df) < 60:
            return 0, 50.0

        pred, conf = self._predict(model, meta, df)
        self._hourly_cache[ticker] = (pred, conf, datetime.now())
        return pred, conf

    def _get_entry(self, ticker: str) -> tuple[int, float, float]:
        """Get 15m entry signal — always fresh."""
        if "15m" not in self.models or ticker not in self.models["15m"]:
            return 0, 50.0, 0.0

        model, meta = self.models["15m"][ticker]
        df = self.fetcher.fetch_indian_stock(ticker, "15m", 59)
        if df is None or len(df) < 60:
            return 0, 50.0, 0.0

        price = float(df["Close"].iloc[-1])
        pred, conf = self._predict(model, meta, df)
        return pred, conf, price

    def _predict(self, model, meta: dict, df: pd.DataFrame) -> tuple[int, float]:
        """Run feature pipeline + model prediction."""
        try:
            features = self.pipeline.transform(df)
            features = features.replace([np.inf, -np.inf], np.nan).dropna()
            if len(features) == 0:
                return 0, 50.0

            trained_feats = meta.get("features", [])
            if trained_feats:
                available = [f for f in trained_feats if f in features.columns]
                if len(available) < len(trained_feats) * 0.8:
                    return 0, 50.0
                features = features[available]

            X = features.iloc[[-1]]
            pred_raw = model.predict(X)[0]
            label_map = {0: -1, 1: 0, 2: 1}
            pred = label_map[int(pred_raw)]

            conf = 50.0
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                conf = float(probs.max()) * 100

                # Override: if BUY/SELL prob > HOLD, use it
                sell_p, hold_p, buy_p = probs[0], probs[1], probs[2]
                if buy_p > hold_p and buy_p > sell_p and buy_p >= 0.30:
                    pred = 1
                    conf = buy_p * 100
                elif sell_p > hold_p and sell_p > buy_p and sell_p >= 0.30:
                    pred = -1
                    conf = sell_p * 100

            return pred, conf
        except Exception as e:
            logger.warning("Prediction failed: %s", e)
            return 0, 50.0

    def _combine(self, regime: str, regime_conf: float,
                 hourly_pred: int, hourly_conf: float,
                 entry_pred: int, entry_conf: float) -> tuple[SignalAction, float]:
        """Apply hierarchical filter rules."""

        # Rule: all must agree for a signal
        if regime == "BULLISH":
            if hourly_pred == 1 and entry_pred == 1:
                return SignalAction.BUY, min(hourly_conf, entry_conf)
            elif hourly_pred == -1:  # Conflict with regime
                return SignalAction.HOLD, 0
        elif regime == "BEARISH":
            if hourly_pred == -1 and entry_pred == -1:
                return SignalAction.SELL, min(hourly_conf, entry_conf)
            elif hourly_pred == 1:  # Conflict with regime
                return SignalAction.HOLD, 0
        else:  # NEUTRAL
            if hourly_pred == 1 and entry_pred == 1:
                return SignalAction.BUY, min(hourly_conf, entry_conf) * 0.8  # Lower conf for neutral regime
            elif hourly_pred == -1 and entry_pred == -1:
                return SignalAction.SELL, min(hourly_conf, entry_conf) * 0.8

        return SignalAction.HOLD, 0
