"""Unit tests for the ML trading pipeline."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv():
    """Generate 200 bars of realistic OHLCV data."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2025-01-01", periods=n, freq="h")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "Open": close + np.random.randn(n) * 0.2,
        "High": close + abs(np.random.randn(n) * 0.5),
        "Low": close - abs(np.random.randn(n) * 0.5),
        "Close": close,
        "Volume": np.random.randint(1000, 100000, n).astype(float),
    }, index=dates)
    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)
    return df


@pytest.fixture
def small_ohlcv():
    """30-bar dataset — too small for the pipeline."""
    np.random.seed(42)
    n = 30
    dates = pd.date_range("2025-01-01", periods=n, freq="h")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "Open": close, "High": close + 1, "Low": close - 1,
        "Close": close, "Volume": np.ones(n) * 10000,
    }, index=dates)


# ── Feature Pipeline Tests ──────────────────────────────────────────

class TestFeaturePipeline:
    def test_transform_produces_valid_output(self, sample_ohlcv):
        from src.ml.feature_pipeline import FeaturePipeline
        pipe = FeaturePipeline()
        features = pipe.transform(sample_ohlcv)

        assert len(features) > 0
        assert not features.isnull().any().any(), "Features contain NaN"
        assert not np.isinf(features.values).any(), "Features contain Inf"

    def test_transform_has_expected_features(self, sample_ohlcv):
        from src.ml.feature_pipeline import FeaturePipeline
        pipe = FeaturePipeline()
        features = pipe.transform(sample_ohlcv)

        expected = ["ema_ratio_8_21", "rsi_14", "atr_pct", "volume_ratio",
                    "body_ratio", "z_score_20", "return_lag_1", "day_of_week"]
        for feat in expected:
            assert feat in features.columns, f"Missing expected feature: {feat}"

    def test_no_look_ahead_bias(self, sample_ohlcv):
        """Features at time t should not depend on data at time t+1."""
        from src.ml.feature_pipeline import FeaturePipeline
        pipe = FeaturePipeline()

        # Get features for first 100 bars
        features_100 = pipe.transform(sample_ohlcv.iloc[:100])
        # Get features for first 150 bars
        features_150 = pipe.transform(sample_ohlcv.iloc[:150])

        # Features for bar 90 should be identical in both
        common_idx = features_100.index.intersection(features_150.index)
        if len(common_idx) > 10:
            idx = common_idx[10]  # Pick a bar well within both
            row_100 = features_100.loc[idx]
            row_150 = features_150.loc[idx]
            # Allow small floating-point differences
            diff = (row_100 - row_150).abs()
            assert (diff < 1e-10).all(), f"Look-ahead bias detected! Diff: {diff[diff >= 1e-10]}"

    def test_insufficient_data_raises(self, small_ohlcv):
        from src.ml.feature_pipeline import FeaturePipeline
        pipe = FeaturePipeline()
        with pytest.raises(ValueError, match="at least 60"):
            pipe.transform(small_ohlcv)

    def test_remove_correlated(self, sample_ohlcv):
        from src.ml.feature_pipeline import FeaturePipeline
        pipe = FeaturePipeline()
        features = pipe.transform(sample_ohlcv)
        n_before = len(features.columns)
        filtered = pipe.remove_correlated(features, threshold=0.95)
        assert len(filtered.columns) <= n_before
        assert len(filtered.columns) > 0

    def test_hour_feature_present_for_intraday(self, sample_ohlcv):
        from src.ml.feature_pipeline import FeaturePipeline
        pipe = FeaturePipeline()
        features = pipe.transform(sample_ohlcv)
        assert "hour" in features.columns, "Hour feature missing for intraday data"


# ── Label Generator Tests ───────────────────────────────────────────

class TestLabelGenerator:
    def test_fixed_threshold_produces_three_classes(self, sample_ohlcv):
        from src.ml.label_generator import LabelGenerator, Label
        gen = LabelGenerator({"round_trip_cost_pct": 0.002, "cost_multiplier": 2.0})
        labels = gen.generate(sample_ohlcv)

        unique = set(labels.dropna().unique())
        assert unique.issubset({Label.BUY, Label.HOLD, Label.SELL})

    def test_last_bar_is_nan(self, sample_ohlcv):
        from src.ml.label_generator import LabelGenerator
        gen = LabelGenerator({"horizon": 1})
        labels = gen.generate(sample_ohlcv)
        assert pd.isna(labels.iloc[-1]), "Last bar should be NaN (no future data)"

    def test_threshold_reflects_cost(self):
        from src.ml.label_generator import LabelGenerator
        gen = LabelGenerator({"round_trip_cost_pct": 0.01, "cost_multiplier": 3.0})
        assert gen.threshold == pytest.approx(0.03)

    def test_class_weights_sum(self, sample_ohlcv):
        from src.ml.label_generator import LabelGenerator
        gen = LabelGenerator({"round_trip_cost_pct": 0.002})
        labels = gen.generate(sample_ohlcv)
        weights = gen.get_class_weights(labels)
        assert len(weights) > 0
        for cls, w in weights.items():
            assert w > 0, f"Weight for class {cls} should be positive"


# ── Cost Model Tests ────────────────────────────────────────────────

class TestCostModel:
    def test_zerodha_intraday_cost(self):
        from src.finance.cost_model import ZerodhaCostModel
        model = ZerodhaCostModel(trade_type="intraday", slippage_bps=0)
        buy = model.calculate(100000, "buy")
        sell = model.calculate(100000, "sell")

        assert buy.total > 0
        assert sell.total > buy.total  # Sell has STT

    def test_zerodha_round_trip(self):
        from src.finance.cost_model import ZerodhaCostModel
        model = ZerodhaCostModel(trade_type="intraday", slippage_bps=10)
        rt_pct = model.round_trip_pct(100000)
        assert 0.001 < rt_pct < 0.01, f"Zerodha intraday round-trip should be 0.1-1%, got {rt_pct*100:.3f}%"

    def test_alpaca_stock_nearly_free(self):
        from src.finance.cost_model import AlpacaCostModel
        model = AlpacaCostModel(asset_type="stock", slippage_bps=0)
        rt_pct = model.round_trip_pct(10000)
        assert rt_pct < 0.001, "Alpaca stocks should be <0.1% round-trip (commission-free)"

    def test_binance_spot_cost(self):
        from src.finance.cost_model import BinanceCostModel
        model = BinanceCostModel(market_type="spot", use_bnb=True, slippage_bps=0)
        rt_pct = model.round_trip_pct(10000)
        assert 0.001 < rt_pct < 0.005, f"Binance spot+BNB should be 0.1-0.5%, got {rt_pct*100:.3f}%"

    def test_factory_function(self):
        from src.finance.cost_model import get_cost_model
        model = get_cost_model("zerodha", trade_type="intraday")
        assert model is not None
        with pytest.raises(ValueError):
            get_cost_model("unknown_broker")

    def test_zero_trade_value(self):
        from src.finance.cost_model import ZerodhaCostModel
        model = ZerodhaCostModel()
        cost = model.calculate(0, "buy")
        assert cost.total == 0


# ── Walk-Forward CV Tests ───────────────────────────────────────────

class TestWalkForwardCV:
    def test_no_overlap_between_train_and_test(self):
        from src.ml.walk_forward import WalkForwardCV
        cv = WalkForwardCV(train_size=50, test_size=20, purge_gap=5)
        folds = cv.split(200)

        for fold in folds:
            # Train end + purge should be <= test start
            assert fold.train_end + 5 <= fold.test_start, \
                f"Train/test overlap in fold {fold.fold_idx}"

    def test_purge_gap_enforced(self):
        from src.ml.walk_forward import WalkForwardCV
        cv = WalkForwardCV(train_size=50, test_size=20, purge_gap=10)
        folds = cv.split(200)

        for fold in folds:
            gap = fold.test_start - fold.train_end
            assert gap >= 10, f"Purge gap {gap} < 10 in fold {fold.fold_idx}"

    def test_holdout_not_in_cv(self):
        from src.ml.walk_forward import WalkForwardCV
        cv = WalkForwardCV(train_size=50, test_size=20, purge_gap=5)
        X = pd.DataFrame(np.random.randn(200, 5))
        y = pd.Series(np.random.choice([0, 1], 200))

        X_cv, y_cv, X_hold, y_hold = cv.get_holdout_split(X, y, 0.15)
        assert len(X_hold) > 0
        assert len(X_cv) + len(X_hold) == len(X)

    def test_auto_configure(self):
        from src.ml.walk_forward import auto_configure_cv
        cv = auto_configure_cv(500, min_folds=3)
        folds = cv.split(int(500 * 0.85))
        assert len(folds) >= 3


# ── Metrics Tests ───────────────────────────────────────────────────

class TestMetrics:
    def test_sharpe_ratio_positive_returns(self):
        from src.ml.metrics import sharpe_ratio
        returns = np.array([0.01, 0.02, 0.015, 0.005, 0.01])
        sr = sharpe_ratio(returns, interval="1d")
        assert sr > 0

    def test_sharpe_ratio_frequency_scaling(self):
        from src.ml.metrics import sharpe_ratio
        returns = np.array([0.001] * 100)
        sr_daily = sharpe_ratio(returns, interval="1d")
        sr_hourly = sharpe_ratio(returns, interval="1h")
        # Hourly should annualize higher (more periods)
        assert sr_hourly > sr_daily

    def test_max_drawdown_negative(self):
        from src.ml.metrics import max_drawdown
        returns = np.array([0.05, -0.10, 0.03, -0.15, 0.02])
        mdd = max_drawdown(returns)
        assert mdd < 0

    def test_profit_factor(self):
        from src.ml.metrics import profit_factor
        returns = np.array([0.05, -0.02, 0.03, -0.01, 0.04])
        pf = profit_factor(returns)
        assert pf > 1  # More profit than loss


# ── Paper Engine Tests ──────────────────────────────────────────────

class TestPaperEngine:
    def test_buy_and_sell_round_trip(self):
        from src.finance.paper_engine import PaperTradingEngine
        from src.finance.base_trader import Signal, SignalAction

        engine = PaperTradingEngine(broker="zerodha", initial_capital=100000, slippage_bps=0)

        buy_signal = Signal(ticker="TEST", action=SignalAction.BUY, confidence=80,
                           price=100, stop_loss=95, take_profit=110)
        engine.execute_signal(buy_signal, 100)
        assert "TEST" in engine.positions

        sell_signal = Signal(ticker="TEST", action=SignalAction.SELL, confidence=80,
                            price=102, stop_loss=105, take_profit=95)
        trade = engine.execute_signal(sell_signal, 102)
        assert trade is not None
        assert "TEST" not in engine.positions

    def test_fees_deducted(self):
        from src.finance.paper_engine import PaperTradingEngine
        from src.finance.base_trader import Signal, SignalAction

        engine = PaperTradingEngine(broker="zerodha", initial_capital=100000, slippage_bps=0)

        buy = Signal(ticker="TEST", action=SignalAction.BUY, confidence=80, price=100)
        engine.execute_signal(buy, 100)

        sell = Signal(ticker="TEST", action=SignalAction.SELL, confidence=80, price=100)
        trade = engine.execute_signal(sell, 100)

        # Buy and sell at same price — should lose money due to fees
        assert engine.capital < 100000, "Fees should reduce capital"
        assert engine.total_fees > 0

    def test_max_positions_enforced(self):
        from src.finance.paper_engine import PaperTradingEngine
        from src.finance.base_trader import Signal, SignalAction

        engine = PaperTradingEngine(broker="zerodha", initial_capital=100000, max_positions=2)

        for ticker in ["A", "B", "C"]:
            sig = Signal(ticker=ticker, action=SignalAction.BUY, confidence=80, price=100)
            engine.execute_signal(sig, 100)

        assert len(engine.positions) == 2, "Should enforce max_positions=2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
