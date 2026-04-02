"""Label generation — creates BUY/HOLD/SELL targets from future returns.

Supports two labeling methods:
1. Fixed-threshold direction classification
2. Triple-barrier labeling (Lopez de Prado)

Thresholds are cost-aware: a trade is only labeled BUY/SELL if the expected
move exceeds the round-trip transaction cost.
"""

import logging
from enum import IntEnum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Label(IntEnum):
    SELL = -1
    HOLD = 0
    BUY = 1


class LabelGenerator:
    """Generates classification labels from OHLCV price data."""

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.method = cfg.get("method", "fixed_threshold")
        self.cost_pct = cfg.get("round_trip_cost_pct", 0.0022)  # Default: Zerodha intraday ~0.22%
        self.cost_multiplier = cfg.get("cost_multiplier", 2.0)  # Label only if move > 2x cost
        self.horizon = cfg.get("horizon", 1)  # Bars to look ahead for return calculation

        # Triple-barrier specific
        self.take_profit_atr = cfg.get("take_profit_atr", 2.0)
        self.stop_loss_atr = cfg.get("stop_loss_atr", 1.5)
        self.max_holding_bars = cfg.get("max_holding_bars", 10)

    @property
    def threshold(self) -> float:
        """Minimum return to classify as BUY or SELL."""
        return self.cost_pct * self.cost_multiplier

    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate labels for the given OHLCV DataFrame.

        Returns a Series of Label values (-1, 0, +1) aligned with the DataFrame index.
        The last `horizon` rows will be NaN (no future data available).
        """
        if self.method == "triple_barrier":
            return self._triple_barrier(df)
        return self._fixed_threshold(df)

    def _fixed_threshold(self, df: pd.DataFrame) -> pd.Series:
        """
        Simple direction classification:
        - BUY if future return > threshold
        - SELL if future return < -threshold
        - HOLD otherwise
        """
        close = df["Close"]
        future_return = close.shift(-self.horizon) / close - 1

        labels = pd.Series(Label.HOLD, index=df.index, dtype=int)
        labels[future_return > self.threshold] = Label.BUY
        labels[future_return < -self.threshold] = Label.SELL

        # NaN out the tail where we can't compute future return
        labels.iloc[-self.horizon:] = np.nan

        dist = labels.dropna().value_counts().to_dict()
        logger.info(
            "Labels (fixed_threshold=%.4f, horizon=%d): BUY=%d, HOLD=%d, SELL=%d",
            self.threshold, self.horizon,
            dist.get(Label.BUY, 0), dist.get(Label.HOLD, 0), dist.get(Label.SELL, 0),
        )
        return labels

    def _triple_barrier(self, df: pd.DataFrame) -> pd.Series:
        """
        Triple-barrier labeling (Lopez de Prado):
        - Upper barrier: take-profit at N * ATR above entry
        - Lower barrier: stop-loss at M * ATR below entry
        - Vertical barrier: max holding period

        The label is determined by which barrier is hit first.
        """
        import pandas_ta as ta

        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values

        atr = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        if atr is None:
            logger.warning("ATR computation failed, falling back to fixed_threshold")
            return self._fixed_threshold(df)
        atr_vals = atr.values

        labels = np.full(len(df), np.nan)

        # Only label bars where we have enough future data
        # Last max_holding_bars remain NaN (no future data to evaluate)
        for i in range(len(df) - self.max_holding_bars):
            if np.isnan(atr_vals[i]) or atr_vals[i] == 0:
                labels[i] = Label.HOLD
                continue

            entry = close[i]
            tp = entry + self.take_profit_atr * atr_vals[i]
            sl = entry - self.stop_loss_atr * atr_vals[i]
            end_idx = i + self.max_holding_bars

            label = Label.HOLD  # Default: vertical barrier hit without tp/sl
            for j in range(i + 1, end_idx + 1):
                if high[j] >= tp:
                    label = Label.BUY
                    break
                if low[j] <= sl:
                    label = Label.SELL
                    break

            # If neither barrier hit, use final return vs cost threshold
            if label == Label.HOLD:
                final_return = (close[end_idx] - entry) / entry
                if final_return > self.threshold:
                    label = Label.BUY
                elif final_return < -self.threshold:
                    label = Label.SELL

            labels[i] = label

        result = pd.Series(labels, index=df.index, dtype="Int64")

        dist = result.dropna().value_counts().to_dict()
        logger.info(
            "Labels (triple_barrier, tp=%.1f*ATR, sl=%.1f*ATR, hold=%d): BUY=%d, HOLD=%d, SELL=%d",
            self.take_profit_atr, self.stop_loss_atr, self.max_holding_bars,
            dist.get(Label.BUY, 0), dist.get(Label.HOLD, 0), dist.get(Label.SELL, 0),
        )
        return result

    def get_class_weights(self, labels: pd.Series) -> dict[int, float]:
        """Compute balanced class weights for imbalanced labels."""
        counts = labels.dropna().value_counts()
        total = counts.sum()
        n_classes = len(counts)
        weights = {}
        for cls, count in counts.items():
            weights[int(cls)] = total / (n_classes * count) if count > 0 else 1.0
        return weights
