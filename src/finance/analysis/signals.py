"""Signal generator — combines technical analysis + sentiment + chart vision."""

import logging
from dataclasses import dataclass

from ..base_trader import Signal, SignalAction
from .technical import TechnicalSignal
from .sentiment import TickerSentiment

logger = logging.getLogger(__name__)


@dataclass
class SignalWeights:
    technical: float = 0.60
    sentiment: float = 0.30
    chart_vision: float = 0.10


class SignalGenerator:
    """Combines multiple analysis sources into a final trading signal."""

    def __init__(self, config: dict | None = None):
        strategy = (config or {}).get("strategy", {})
        weights = strategy.get("signal_weights", {})
        self.weights = SignalWeights(
            technical=weights.get("technical", 0.60),
            sentiment=weights.get("sentiment", 0.30),
            chart_vision=weights.get("chart_vision", 0.10),
        )
        self.threshold = strategy.get("signal_threshold", 70)

    def generate(
        self,
        ticker: str,
        tech_signal: TechnicalSignal | None = None,
        sentiment: TickerSentiment | None = None,
        chart_analysis: dict | None = None,
    ) -> Signal:
        """
        Generate a final signal by combining all sources.
        Each source produces a score from -100 (strong sell) to +100 (strong buy).
        """
        # Convert technical signal to -100 to +100 scale
        tech_score = 0.0
        if tech_signal:
            if tech_signal.action == "BUY":
                tech_score = tech_signal.confidence
            elif tech_signal.action == "SELL":
                tech_score = -tech_signal.confidence
            # HOLD stays 0

        # Convert sentiment (-5 to +5) to -100 to +100 scale
        sent_score = 0.0
        if sentiment:
            sent_score = sentiment.score * 20  # -5..+5 -> -100..+100
            # Adjust by confidence
            sent_score *= (sentiment.confidence / 100)

        # Convert chart vision to -100 to +100 scale
        chart_score = 0.0
        if chart_analysis:
            rec = chart_analysis.get("recommendation", "HOLD")
            conf = chart_analysis.get("confidence", 50)
            if rec == "BUY":
                chart_score = conf
            elif rec == "SELL":
                chart_score = -conf

        # Weighted combination
        total_weight = 0.0
        weighted_score = 0.0

        if tech_signal:
            weighted_score += tech_score * self.weights.technical
            total_weight += self.weights.technical
        if sentiment:
            weighted_score += sent_score * self.weights.sentiment
            total_weight += self.weights.sentiment
        if chart_analysis:
            weighted_score += chart_score * self.weights.chart_vision
            total_weight += self.weights.chart_vision

        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0

        # Determine action
        confidence = abs(final_score)
        if final_score > 0 and confidence >= self.threshold:
            action = SignalAction.BUY
        elif final_score < 0 and confidence >= self.threshold:
            action = SignalAction.SELL
        else:
            action = SignalAction.HOLD

        # Build reasoning
        reasons = []
        if tech_signal and tech_signal.reasons:
            reasons.append(f"Technical: {', '.join(tech_signal.reasons[:3])}")
        if sentiment:
            reasons.append(f"Sentiment: {sentiment.score:+.1f}/5 ({sentiment.article_count} articles)")
        if chart_analysis and chart_analysis.get("reasoning"):
            reasons.append(f"Chart: {chart_analysis['reasoning'][:100]}")

        price = 0.0
        stop_loss = None
        take_profit = None
        if tech_signal and tech_signal.indicators:
            price = tech_signal.indicators.get("current_price", 0)
            stop_loss = tech_signal.indicators.get("suggested_stop_loss")
            take_profit = tech_signal.indicators.get("suggested_take_profit")

        signal = Signal(
            ticker=ticker,
            action=action,
            confidence=round(confidence, 1),
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=" | ".join(reasons),
            technical_score=round(tech_score, 1),
            sentiment_score=round(sent_score, 1),
            chart_score=round(chart_score, 1),
        )

        logger.info(
            "Signal %s: %s (conf=%.1f%%) tech=%.1f sent=%.1f chart=%.1f",
            ticker, action.value, confidence, tech_score, sent_score, chart_score,
        )
        return signal
