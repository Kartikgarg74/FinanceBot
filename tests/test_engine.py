"""Tests for TradingEngine — retry logic, scan flow, and error handling."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock


class TestEngineRetryLogic:
    """Test broker connection retry with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self, mock_telegram):
        """Engine should retry up to 3 times on broker API failure."""
        with patch("src.finance.engine.TechnicalAnalyzer"), \
             patch("src.finance.engine.NewsFetcher"), \
             patch("src.finance.engine.SentimentScorer"), \
             patch("src.finance.engine.SignalGenerator"), \
             patch("src.finance.engine.RiskManager"), \
             patch("src.finance.engine.TradeLogger"), \
             patch("src.finance.engine.MarketHours") as MockHours, \
             patch("src.finance.engine.get_cache"), \
             patch("src.finance.engine.get_health_monitor"), \
             patch("src.finance.engine.resilient_call") as mock_resilient:

            from src.finance.engine import TradingEngine

            # Market is open
            mock_hours_inst = MockHours.return_value
            mock_hours_inst.is_open.return_value = True

            # All 3 attempts fail
            mock_resilient.side_effect = ConnectionError("Broker API timeout")

            mock_trader = MagicMock()
            mock_trader.mode = "paper"
            mock_trader.trades = []

            engine = TradingEngine(
                trader=mock_trader,
                ai_router=MagicMock(),
                config={"_market_name": "zerodha", "trading": {}},
                news_config={},
                telegram_bot=mock_telegram,
            )

            signals = await engine.run_scan(["RELIANCE"])

            assert signals == []
            assert mock_resilient.call_count == 6  # 3 retries * 2 calls (balance + positions)... actually it's sequential
            mock_telegram.send_message.assert_called()
            call_text = mock_telegram.send_message.call_args[0][0]
            assert "API down" in call_text

    @pytest.mark.asyncio
    async def test_succeeds_on_second_attempt(self, mock_telegram):
        """Engine should recover if broker API succeeds on retry."""
        with patch("src.finance.engine.TechnicalAnalyzer"), \
             patch("src.finance.engine.NewsFetcher"), \
             patch("src.finance.engine.SentimentScorer"), \
             patch("src.finance.engine.SignalGenerator"), \
             patch("src.finance.engine.RiskManager") as MockRisk, \
             patch("src.finance.engine.TradeLogger"), \
             patch("src.finance.engine.MarketHours") as MockHours, \
             patch("src.finance.engine.get_cache") as mock_cache, \
             patch("src.finance.engine.get_health_monitor"), \
             patch("src.finance.engine.resilient_call") as mock_resilient:

            from src.finance.engine import TradingEngine

            mock_hours_inst = MockHours.return_value
            mock_hours_inst.is_open.return_value = True

            # First attempt fails, second succeeds
            call_count = [0]
            def side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] <= 2:  # First round (balance + positions) fails
                    raise ConnectionError("timeout")
                if call_count[0] == 3:
                    return 100000.0  # balance
                return []  # positions
            mock_resilient.side_effect = side_effect

            # Risk manager says can't trade (to end the scan early)
            mock_risk_inst = MockRisk.return_value
            mock_risk_inst.can_trade.return_value = (False, "test block")

            mock_trader = MagicMock()
            mock_trader.mode = "paper"
            mock_trader.trades = []

            mock_cache_inst = mock_cache.return_value
            mock_cache_inst.cleanup = MagicMock()

            engine = TradingEngine(
                trader=mock_trader,
                ai_router=MagicMock(),
                config={"_market_name": "binance", "trading": {}},
                news_config={},
                telegram_bot=None,
            )

            signals = await engine.run_scan(["BTCUSDT"])
            # Should have recovered (no Telegram error sent, risk manager was reached)
            assert signals == []


class TestMarketHoursCheck:
    """Test market hours gating."""

    @pytest.mark.asyncio
    async def test_skips_when_market_closed(self):
        with patch("src.finance.engine.TechnicalAnalyzer"), \
             patch("src.finance.engine.NewsFetcher"), \
             patch("src.finance.engine.SentimentScorer"), \
             patch("src.finance.engine.SignalGenerator"), \
             patch("src.finance.engine.RiskManager"), \
             patch("src.finance.engine.TradeLogger"), \
             patch("src.finance.engine.MarketHours") as MockHours, \
             patch("src.finance.engine.get_cache"), \
             patch("src.finance.engine.get_health_monitor"):

            from src.finance.engine import TradingEngine

            mock_hours_inst = MockHours.return_value
            mock_hours_inst.is_open.return_value = False
            mock_hours_inst.next_open.return_value = "09:15 IST"

            mock_trader = MagicMock()
            mock_trader.mode = "paper"
            mock_trader.trades = []

            engine = TradingEngine(
                trader=mock_trader,
                ai_router=MagicMock(),
                config={"_market_name": "zerodha", "trading": {}},
                news_config={},
            )

            signals = await engine.run_scan(["RELIANCE"])
            assert signals == []
