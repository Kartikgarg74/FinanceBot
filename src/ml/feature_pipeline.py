"""Feature engineering pipeline — converts raw OHLCV into ML-ready features.

All features use .shift(1) where applicable to prevent look-ahead bias.
The model only sees information available BEFORE the current bar closes.
"""

import logging

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Converts raw OHLCV DataFrame into a feature matrix for ML models."""

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.feature_groups = cfg.get("feature_groups", [
            "trend", "momentum", "volatility", "volume",
            "price_action", "statistical", "lag", "time",
        ])
        self.drop_na = cfg.get("drop_na", True)
        self.max_correlation = cfg.get("max_correlation", 0.95)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point. Takes OHLCV DataFrame, returns feature DataFrame.

        Input columns: Open, High, Low, Close, Volume
        Output: DataFrame with all engineered features (NaN rows dropped).
        """
        if df is None or len(df) < 60:
            raise ValueError(f"Need at least 60 bars, got {len(df) if df is not None else 0}")

        df = df.copy()
        features = pd.DataFrame(index=df.index)

        group_builders = {
            "trend": self._trend_features,
            "momentum": self._momentum_features,
            "volatility": self._volatility_features,
            "volume": self._volume_features,
            "price_action": self._price_action_features,
            "statistical": self._statistical_features,
            "lag": self._lag_features,
            "time": self._time_features,
        }

        for group in self.feature_groups:
            builder = group_builders.get(group)
            if builder:
                group_feats = builder(df)
                features = pd.concat([features, group_feats], axis=1)

        if self.drop_na:
            features = features.dropna()

        # Replace any remaining inf values with NaN, then drop
        features = features.replace([np.inf, -np.inf], np.nan).dropna()

        logger.info("Feature pipeline: %d bars → %d samples, %d features",
                     len(df), len(features), len(features.columns))
        return features

    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        """Return the list of feature column names without transforming."""
        return list(self.transform(df).columns)

    # ------------------------------------------------------------------
    # Feature group builders
    # ------------------------------------------------------------------

    def _trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """EMA ratios and price-to-EMA distances."""
        f = pd.DataFrame(index=df.index)
        close = df["Close"]

        ema8 = ta.ema(close, length=8)
        ema21 = ta.ema(close, length=21)
        ema50 = ta.ema(close, length=50)

        # Ratios (normalized crossover signals) — shifted to avoid look-ahead
        f["ema_ratio_8_21"] = (ema8 / ema21).shift(1)
        f["ema_ratio_21_50"] = (ema21 / ema50).shift(1)

        # Price distance from EMAs (percentage)
        f["price_to_ema21"] = (close / ema21 - 1).shift(1)
        f["price_to_ema50"] = (close / ema50 - 1).shift(1)

        return f

    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI, MACD, Rate of Change."""
        f = pd.DataFrame(index=df.index)
        close = df["Close"]

        # RSI at multiple timeframes
        rsi14 = ta.rsi(close, length=14)
        f["rsi_14"] = rsi14.shift(1)
        f["rsi_14_slope"] = rsi14.diff(3).shift(1)  # 3-bar rate of RSI change

        # MACD
        macd_result = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_result is not None and len(macd_result.columns) >= 3:
            macd_hist = macd_result.iloc[:, 1]  # Histogram
            f["macd_histogram"] = macd_hist.shift(1)
            f["macd_hist_slope"] = macd_hist.diff(2).shift(1)

        # Rate of Change
        f["roc_5"] = ta.roc(close, length=5).shift(1)
        f["roc_10"] = ta.roc(close, length=10).shift(1)

        # Stochastic %K
        stoch = ta.stoch(df["High"], df["Low"], close, k=14, d=3)
        if stoch is not None and len(stoch.columns) >= 2:
            f["stoch_k"] = stoch.iloc[:, 0].shift(1)

        return f

    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR, Bollinger Bands, historical volatility."""
        f = pd.DataFrame(index=df.index)
        close = df["Close"]

        # ATR normalized by price
        atr14 = ta.atr(df["High"], df["Low"], close, length=14)
        f["atr_pct"] = (atr14 / close).shift(1)

        # Bollinger Bands %B (position within bands)
        bb = ta.bbands(close, length=20, std=2)
        if bb is not None and len(bb.columns) >= 3:
            bb_lower = bb.iloc[:, 0]
            bb_upper = bb.iloc[:, 2]
            bb_width = bb_upper - bb_lower
            f["bb_pct_b"] = ((close - bb_lower) / bb_width.replace(0, np.nan)).shift(1)
            f["bb_width_pct"] = (bb_width / close).shift(1)

        # Rolling realized volatility (log returns std)
        log_ret = np.log(close / close.shift(1))
        f["vol_10"] = log_ret.rolling(10).std().shift(1)
        f["vol_20"] = log_ret.rolling(20).std().shift(1)
        f["vol_ratio"] = (f["vol_10"] / f["vol_20"].replace(0, np.nan))

        return f

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume profile and money flow."""
        f = pd.DataFrame(index=df.index)

        if "Volume" not in df.columns or df["Volume"].sum() == 0:
            return f

        vol = df["Volume"]
        close = df["Close"]

        # Volume ratio vs 20-period average
        vol_sma20 = vol.rolling(20).mean()
        f["volume_ratio"] = (vol / vol_sma20.replace(0, np.nan)).shift(1)

        # OBV slope (On-Balance Volume rate of change)
        obv = ta.obv(close, vol)
        if obv is not None:
            f["obv_slope"] = obv.diff(5).shift(1) / (obv.abs().rolling(20).mean().replace(0, np.nan))

        # Money Flow Index
        mfi = ta.mfi(df["High"], df["Low"], close, vol, length=14)
        if mfi is not None:
            f["mfi_14"] = mfi.shift(1)

        # Volume-price divergence: price up but volume down (or vice versa)
        price_change = close.pct_change(5)
        vol_change = vol.pct_change(5)
        f["vol_price_divergence"] = (price_change * vol_change).shift(1)

        return f

    def _price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Candlestick body/shadow ratios, gaps."""
        f = pd.DataFrame(index=df.index)
        o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]

        candle_range = (h - l).replace(0, np.nan)

        # Body size as fraction of range
        f["body_ratio"] = (abs(c - o) / candle_range).shift(1)

        # Upper and lower shadows as fraction of range
        f["upper_shadow"] = ((h - pd.concat([o, c], axis=1).max(axis=1)) / candle_range).shift(1)
        f["lower_shadow"] = ((pd.concat([o, c], axis=1).min(axis=1) - l) / candle_range).shift(1)

        # Gap (open vs previous close)
        f["gap_pct"] = (o / c.shift(1) - 1).shift(1)

        # Distance from recent high/low
        f["pct_from_20d_high"] = (c / h.rolling(20).max() - 1).shift(1)
        f["pct_from_20d_low"] = (c / l.rolling(20).min() - 1).shift(1)

        return f

    def _statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score, skewness, kurtosis of returns."""
        f = pd.DataFrame(index=df.index)
        close = df["Close"]

        log_ret = np.log(close / close.shift(1))

        # Z-score of price relative to 20-day rolling mean/std
        roll_mean = close.rolling(20).mean()
        roll_std = close.rolling(20).std()
        f["z_score_20"] = ((close - roll_mean) / roll_std.replace(0, np.nan)).shift(1)

        # Rolling skewness and kurtosis of returns
        f["skew_20"] = log_ret.rolling(20).skew().shift(1)
        f["kurt_20"] = log_ret.rolling(20).kurt().shift(1)

        return f

    def _lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lagged returns at various horizons."""
        f = pd.DataFrame(index=df.index)
        close = df["Close"]

        # Log returns at different lags — these are already "past" so shift(1) on
        # the 1-bar return, but multi-bar returns inherently look back
        log_ret = np.log(close / close.shift(1))

        f["return_lag_1"] = log_ret.shift(1)
        f["return_lag_2"] = log_ret.shift(2)
        f["return_lag_3"] = log_ret.shift(3)
        f["return_lag_5"] = np.log(close / close.shift(5)).shift(1)
        f["return_lag_10"] = np.log(close / close.shift(10)).shift(1)

        return f

    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calendar-based features."""
        f = pd.DataFrame(index=df.index)

        idx = df.index
        if hasattr(idx, "dayofweek"):
            f["day_of_week"] = idx.dayofweek
        elif hasattr(idx, "to_series"):
            f["day_of_week"] = idx.to_series().dt.dayofweek

        if hasattr(idx, "hour"):
            hour = idx.hour
            # Only add if intraday data (not all zeros)
            if hour.max() > 0:
                f["hour"] = hour

        return f

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def remove_correlated(self, features: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
        """Remove features with pairwise correlation above threshold."""
        thresh = threshold or self.max_correlation
        clean = features.dropna()
        if clean.empty:
            logger.warning("No valid samples for correlation analysis")
            return features
        corr = clean.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > thresh)]
        if to_drop:
            logger.info("Dropping %d correlated features: %s", len(to_drop), to_drop)
        return features.drop(columns=to_drop, errors="ignore")

    def normalize_rolling(self, features: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        """Rolling z-score normalization (no look-ahead bias)."""
        roll_mean = features.rolling(window, min_periods=window).mean()
        roll_std = features.rolling(window, min_periods=window).std()
        normalized = (features - roll_mean) / roll_std.replace(0, np.nan)
        return normalized.dropna()
