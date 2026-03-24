"""Custom FreqTrade strategy for crypto trading (KartikAICryptoStrategy).

This file can be used directly with FreqTrade:
  freqtrade trade --strategy KartikAICryptoStrategy --config config/freqtrade.json

For standalone use (without FreqTrade), use BinanceClient instead.
"""

import logging

logger = logging.getLogger(__name__)

# FreqTrade strategy — only importable if freqtrade is installed
try:
    import numpy as np
    import pandas as pd
    import talib.abstract as ta
    from freqtrade.strategy import IStrategy, merge_informative_pair
    from freqtrade.strategy import DecimalParameter, IntParameter

    class KartikAICryptoStrategy(IStrategy):
        """
        Multi-timeframe strategy with ML overlay.
        - 5m primary + 15m/1h/4h confirmation
        - RSI, MACD, Bollinger, EMA, Volume
        - Dynamic stoploss based on ATR
        """

        INTERFACE_VERSION = 3
        timeframe = "5m"
        informative_timeframes = ["15m", "1h", "4h"]

        # Stoploss
        stoploss = -0.05
        trailing_stop = True
        trailing_stop_positive = 0.01
        trailing_stop_positive_offset = 0.03
        trailing_only_offset_is_reached = True

        # ROI
        minimal_roi = {
            "0": 0.08,    # 8% profit target
            "30": 0.04,   # 4% after 30 min
            "60": 0.02,   # 2% after 1 hour
            "120": 0.01,  # 1% after 2 hours
        }

        # Hyperopt parameters
        buy_rsi = IntParameter(20, 40, default=30, space="buy")
        buy_ema_short = IntParameter(10, 30, default=20, space="buy")
        buy_ema_long = IntParameter(40, 100, default=50, space="buy")
        sell_rsi = IntParameter(60, 85, default=70, space="sell")

        def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
            """Calculate all technical indicators."""
            # RSI
            dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

            # MACD
            macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
            dataframe["macd"] = macd["macd"]
            dataframe["macdsignal"] = macd["macdsignal"]
            dataframe["macdhist"] = macd["macdhist"]

            # EMAs
            dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
            dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
            dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)

            # Bollinger Bands
            bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
            dataframe["bb_upper"] = bb["upperband"]
            dataframe["bb_mid"] = bb["middleband"]
            dataframe["bb_lower"] = bb["lowerband"]

            # ATR for dynamic stoploss
            dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

            # Volume SMA
            dataframe["volume_sma"] = dataframe["volume"].rolling(20).mean()

            # Stochastic
            stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
            dataframe["stoch_k"] = stoch["slowk"]
            dataframe["stoch_d"] = stoch["slowd"]

            return dataframe

        def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
            """Define buy conditions."""
            dataframe.loc[
                (
                    # RSI oversold
                    (dataframe["rsi"] < self.buy_rsi.value) &
                    # Price below lower Bollinger
                    (dataframe["close"] <= dataframe["bb_lower"]) &
                    # MACD bullish crossover
                    (dataframe["macd"] > dataframe["macdsignal"]) &
                    # EMA alignment (short above long)
                    (dataframe["ema_20"] > dataframe["ema_50"]) &
                    # Volume above average
                    (dataframe["volume"] > dataframe["volume_sma"]) &
                    # Basic candle filter
                    (dataframe["volume"] > 0)
                ),
                "enter_long",
            ] = 1

            return dataframe

        def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
            """Define sell conditions."""
            dataframe.loc[
                (
                    # RSI overbought
                    (dataframe["rsi"] > self.sell_rsi.value) &
                    # Price above upper Bollinger
                    (dataframe["close"] >= dataframe["bb_upper"]) &
                    # MACD bearish crossover
                    (dataframe["macd"] < dataframe["macdsignal"]) &
                    # Volume confirmation
                    (dataframe["volume"] > 0)
                ),
                "exit_long",
            ] = 1

            return dataframe

        def custom_stoploss(self, pair: str, trade, current_time,
                            current_rate, current_profit, **kwargs) -> float:
            """Dynamic stoploss based on ATR."""
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is not None and len(dataframe) > 0:
                last_candle = dataframe.iloc[-1]
                atr = last_candle.get("atr", 0)
                if atr > 0 and current_rate > 0:
                    # Stop loss at 2x ATR below current price
                    sl_distance = (2 * atr) / current_rate
                    return -sl_distance
            return self.stoploss

except ImportError:
    logger.debug("FreqTrade not installed — KartikAICryptoStrategy unavailable. Use BinanceClient instead.")
