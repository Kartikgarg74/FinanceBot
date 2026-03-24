"""Technical analysis engine — calculates indicators and generates signals."""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


@dataclass
class TechnicalSignal:
    """Result of technical analysis for a single ticker."""
    ticker: str
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0-100
    indicators: dict  # Raw indicator values
    reasons: list[str]  # Human-readable reasons


class TechnicalAnalyzer:
    """Calculates technical indicators and generates buy/sell signals."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        strategy = self.config.get("strategy", {})
        self.signal_threshold = strategy.get("signal_threshold", 70)

    def analyze(self, df: pd.DataFrame, ticker: str = "") -> TechnicalSignal:
        """
        Run full technical analysis on OHLCV DataFrame.
        Expected columns: Open, High, Low, Close, Volume
        """
        if df is None or len(df) < 50:
            return TechnicalSignal(ticker=ticker, action="HOLD", confidence=0,
                                    indicators={}, reasons=["Insufficient data"])

        df = df.copy()
        indicators = {}
        buy_signals = 0
        sell_signals = 0
        reasons = []
        total_weight = 0

        # --- RSI ---
        df["RSI"] = ta.rsi(df["Close"], length=14)
        rsi = df["RSI"].iloc[-1] if not pd.isna(df["RSI"].iloc[-1]) else 50
        indicators["RSI"] = round(rsi, 2)
        if rsi < 30:
            buy_signals += 2
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi < 40:
            buy_signals += 1
            reasons.append(f"RSI approaching oversold ({rsi:.1f})")
        elif rsi > 70:
            sell_signals += 2
            reasons.append(f"RSI overbought ({rsi:.1f})")
        elif rsi > 60:
            sell_signals += 1
            reasons.append(f"RSI approaching overbought ({rsi:.1f})")
        total_weight += 2

        # --- MACD ---
        macd_result = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        if macd_result is not None and len(macd_result.columns) >= 3:
            df = pd.concat([df, macd_result], axis=1)
            macd_line = macd_result.iloc[-1, 0]
            signal_line = macd_result.iloc[-1, 2]
            macd_prev = macd_result.iloc[-2, 0] if len(macd_result) > 1 else 0
            signal_prev = macd_result.iloc[-2, 2] if len(macd_result) > 1 else 0

            if not pd.isna(macd_line) and not pd.isna(signal_line):
                indicators["MACD"] = round(macd_line, 4)
                indicators["MACD_signal"] = round(signal_line, 4)
                # Bullish crossover
                if macd_prev < signal_prev and macd_line > signal_line:
                    buy_signals += 2
                    reasons.append("MACD bullish crossover")
                # Bearish crossover
                elif macd_prev > signal_prev and macd_line < signal_line:
                    sell_signals += 2
                    reasons.append("MACD bearish crossover")
                elif macd_line > signal_line:
                    buy_signals += 1
                elif macd_line < signal_line:
                    sell_signals += 1
        total_weight += 2

        # --- EMA (20, 50, 200) ---
        for period in [20, 50, 200]:
            col = f"EMA_{period}"
            df[col] = ta.ema(df["Close"], length=period)
            val = df[col].iloc[-1] if not pd.isna(df[col].iloc[-1]) else None
            if val:
                indicators[col] = round(val, 2)

        price = df["Close"].iloc[-1]
        ema20 = indicators.get("EMA_20")
        ema50 = indicators.get("EMA_50")
        ema200 = indicators.get("EMA_200")

        if ema20 and ema50:
            if price > ema20 > ema50:
                buy_signals += 2
                reasons.append("Price above EMA20 > EMA50 (bullish alignment)")
            elif price < ema20 < ema50:
                sell_signals += 2
                reasons.append("Price below EMA20 < EMA50 (bearish alignment)")
        total_weight += 2

        if ema200:
            if price > ema200:
                buy_signals += 1
                reasons.append("Price above EMA200 (long-term bullish)")
            else:
                sell_signals += 1
                reasons.append("Price below EMA200 (long-term bearish)")
        total_weight += 1

        # --- Bollinger Bands ---
        bb = ta.bbands(df["Close"], length=20, std=2)
        if bb is not None and len(bb.columns) >= 3:
            bb_lower = bb.iloc[-1, 0]
            bb_mid = bb.iloc[-1, 1]
            bb_upper = bb.iloc[-1, 2]
            if not pd.isna(bb_lower):
                indicators["BB_lower"] = round(bb_lower, 2)
                indicators["BB_mid"] = round(bb_mid, 2)
                indicators["BB_upper"] = round(bb_upper, 2)
                if price <= bb_lower:
                    buy_signals += 2
                    reasons.append(f"Price at lower Bollinger Band ({price:.2f} <= {bb_lower:.2f})")
                elif price >= bb_upper:
                    sell_signals += 2
                    reasons.append(f"Price at upper Bollinger Band ({price:.2f} >= {bb_upper:.2f})")
        total_weight += 2

        # --- ATR (for stop-loss calculation) ---
        df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        atr = df["ATR"].iloc[-1] if not pd.isna(df["ATR"].iloc[-1]) else 0
        indicators["ATR"] = round(atr, 2)

        # --- Volume analysis ---
        if "Volume" in df.columns:
            vol_sma = df["Volume"].rolling(20).mean().iloc[-1]
            current_vol = df["Volume"].iloc[-1]
            if vol_sma and current_vol and vol_sma > 0:
                vol_ratio = current_vol / vol_sma
                indicators["volume_ratio"] = round(vol_ratio, 2)
                if vol_ratio > 1.5:
                    reasons.append(f"High volume ({vol_ratio:.1f}x avg)")

        # --- Calculate final signal ---
        if total_weight == 0:
            return TechnicalSignal(ticker=ticker, action="HOLD", confidence=0,
                                    indicators=indicators, reasons=reasons)

        net_score = (buy_signals - sell_signals) / total_weight
        confidence = min(abs(net_score) * 100, 100)

        if net_score > 0.2 and confidence >= 30:
            action = "BUY"
        elif net_score < -0.2 and confidence >= 30:
            action = "SELL"
        else:
            action = "HOLD"

        indicators["current_price"] = round(price, 2)
        indicators["suggested_stop_loss"] = round(price - 2 * atr, 2) if atr else None
        indicators["suggested_take_profit"] = round(price + 3 * atr, 2) if atr else None

        return TechnicalSignal(
            ticker=ticker,
            action=action,
            confidence=round(confidence, 1),
            indicators=indicators,
            reasons=reasons,
        )
