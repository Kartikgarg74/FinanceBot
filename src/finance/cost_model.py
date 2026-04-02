"""Transaction cost models for each supported broker.

Calculates exact fees + slippage for realistic backtesting and
cost-aware label generation.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TradeCost:
    """Breakdown of a single trade's costs."""
    brokerage: float = 0.0
    stt: float = 0.0
    transaction_charges: float = 0.0
    gst: float = 0.0
    sebi_charges: float = 0.0
    stamp_duty: float = 0.0
    slippage: float = 0.0

    @property
    def total(self) -> float:
        return (self.brokerage + self.stt + self.transaction_charges +
                self.gst + self.sebi_charges + self.stamp_duty + self.slippage)

    @property
    def total_pct(self) -> float:
        """For display purposes — needs to be computed relative to trade value."""
        return self.total


class ZerodhaCostModel:
    """Zerodha fee structure (as of October 2024 revision)."""

    def __init__(self, trade_type: str = "intraday", slippage_bps: float = 10):
        """
        trade_type: 'intraday' (MIS), 'delivery' (CNC), 'futures', 'options'
        slippage_bps: assumed slippage in basis points
        """
        self.trade_type = trade_type
        self.slippage_bps = slippage_bps

    def calculate(self, trade_value: float, side: str = "buy") -> TradeCost:
        """Calculate fees for a single leg (buy or sell)."""
        cost = TradeCost()

        if self.trade_type == "intraday":
            cost.brokerage = min(trade_value * 0.0003, 20)  # 0.03% or Rs 20
            if side == "sell":
                cost.stt = trade_value * 0.000250  # 0.025% on sell
            cost.transaction_charges = trade_value * 0.0000297  # NSE
            cost.stamp_duty = trade_value * 0.00003 if side == "buy" else 0  # 0.003% buy

        elif self.trade_type == "delivery":
            cost.brokerage = 0  # Free delivery on Zerodha
            cost.stt = trade_value * 0.001  # 0.1% on both sides
            cost.transaction_charges = trade_value * 0.0000297
            cost.stamp_duty = trade_value * 0.00015 if side == "buy" else 0  # 0.015% buy

        elif self.trade_type == "futures":
            cost.brokerage = 20  # Flat Rs 20
            if side == "sell":
                cost.stt = trade_value * 0.000125  # 0.0125% on sell
            cost.transaction_charges = trade_value * 0.0000173

        elif self.trade_type == "options":
            cost.brokerage = 20  # Flat Rs 20
            if side == "sell":
                cost.stt = trade_value * 0.000625  # 0.0625% on sell (premium)
            cost.transaction_charges = trade_value * 0.000495

        # GST: 18% on (brokerage + transaction charges)
        cost.gst = (cost.brokerage + cost.transaction_charges) * 0.18

        # SEBI charges: Rs 10 per crore
        cost.sebi_charges = trade_value * 0.000001  # 0.0001%

        # Slippage
        cost.slippage = trade_value * (self.slippage_bps / 10000)

        return cost

    def round_trip_cost(self, trade_value: float) -> float:
        """Total cost for buy + sell (round trip)."""
        buy_cost = self.calculate(trade_value, "buy")
        sell_cost = self.calculate(trade_value, "sell")
        return buy_cost.total + sell_cost.total

    def round_trip_pct(self, trade_value: float) -> float:
        """Round-trip cost as a percentage of trade value."""
        return self.round_trip_cost(trade_value) / trade_value if trade_value > 0 else 0


class AlpacaCostModel:
    """Alpaca (US) fee structure — commission-free stocks."""

    def __init__(self, asset_type: str = "stock", slippage_bps: float = 2):
        """
        asset_type: 'stock' or 'crypto'
        """
        self.asset_type = asset_type
        self.slippage_bps = slippage_bps

    def calculate(self, trade_value: float, side: str = "buy") -> TradeCost:
        cost = TradeCost()

        if self.asset_type == "stock":
            cost.brokerage = 0  # Commission-free
            # SEC fee on sell only (~$24.90 per million as of 2024)
            if side == "sell":
                cost.transaction_charges = trade_value * 0.0000249
            # FINRA TAF: ~$0.000119 per share — approximate as fraction of value
            cost.sebi_charges = trade_value * 0.0000013  # ~0.00013% approximation

        elif self.asset_type == "crypto":
            cost.brokerage = trade_value * 0.0025  # 0.25% per trade

        cost.slippage = trade_value * (self.slippage_bps / 10000)
        return cost

    def round_trip_cost(self, trade_value: float) -> float:
        buy = self.calculate(trade_value, "buy")
        sell = self.calculate(trade_value, "sell")
        return buy.total + sell.total

    def round_trip_pct(self, trade_value: float) -> float:
        return self.round_trip_cost(trade_value) / trade_value if trade_value > 0 else 0


class BinanceCostModel:
    """Binance fee structure for spot and futures."""

    def __init__(self, market_type: str = "spot", use_bnb: bool = True, slippage_bps: float = 5):
        """
        market_type: 'spot' or 'futures'
        use_bnb: whether BNB discount applies (25% off)
        """
        self.market_type = market_type
        self.use_bnb = use_bnb
        self.slippage_bps = slippage_bps

    def calculate(self, trade_value: float, side: str = "buy") -> TradeCost:
        cost = TradeCost()

        if self.market_type == "spot":
            base_fee = 0.001  # 0.10% maker/taker
            if self.use_bnb:
                base_fee *= 0.75  # 25% discount
            cost.brokerage = trade_value * base_fee

        elif self.market_type == "futures":
            if side == "buy":  # Assume taker for market orders
                cost.brokerage = trade_value * 0.0005  # 0.05% taker
            else:
                cost.brokerage = trade_value * 0.0002  # 0.02% maker
            if self.use_bnb:
                cost.brokerage *= 0.90  # 10% BNB discount on futures

        cost.slippage = trade_value * (self.slippage_bps / 10000)
        return cost

    def round_trip_cost(self, trade_value: float) -> float:
        buy = self.calculate(trade_value, "buy")
        sell = self.calculate(trade_value, "sell")
        return buy.total + sell.total

    def round_trip_pct(self, trade_value: float) -> float:
        return self.round_trip_cost(trade_value) / trade_value if trade_value > 0 else 0


def get_cost_model(broker: str, **kwargs):
    """Factory for cost models.

    Normalizes parameter names across brokers:
    - 'trade_type' maps to Zerodha's trade_type, Alpaca's asset_type, Binance's market_type
    """
    models = {
        "zerodha": ZerodhaCostModel,
        "alpaca": AlpacaCostModel,
        "binance": BinanceCostModel,
    }
    cls = models.get(broker.lower())
    if cls is None:
        raise ValueError(f"Unknown broker: {broker}. Available: {list(models.keys())}")

    # Normalize trade_type param to broker-specific name
    trade_type = kwargs.pop("trade_type", None)
    if trade_type and broker.lower() == "alpaca":
        kwargs["asset_type"] = trade_type
    elif trade_type and broker.lower() == "binance":
        kwargs["market_type"] = trade_type
    elif trade_type and broker.lower() == "zerodha":
        kwargs["trade_type"] = trade_type

    return cls(**kwargs)


def estimate_round_trip_pct(broker: str, trade_value: float = 100000, **kwargs) -> float:
    """Quick helper: get round-trip cost as percentage for a given broker."""
    model = get_cost_model(broker, **kwargs)
    return model.round_trip_pct(trade_value)
