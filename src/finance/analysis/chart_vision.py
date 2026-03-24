"""Chart vision module — generates candlestick charts and analyzes via Claude Vision."""

import base64
import io
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ChartVisionAnalyzer:
    """Generates candlestick charts and sends to Claude Vision for pattern recognition."""

    def __init__(self, ai_router, output_dir: str = "data/charts"):
        self.ai = ai_router
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_chart_image(
        self,
        df: pd.DataFrame,
        ticker: str,
        days: int = 60,
    ) -> bytes | None:
        """Generate a candlestick chart with indicators as a PNG image."""
        try:
            import mplfinance as mpf
            import pandas_ta as ta

            df = df.copy().tail(days)

            if len(df) < 10:
                logger.warning("Not enough data to generate chart for %s", ticker)
                return None

            # Ensure proper column names for mplfinance
            required = ["Open", "High", "Low", "Close", "Volume"]
            for col in required:
                if col not in df.columns:
                    logger.error("Missing column %s for chart generation", col)
                    return None

            # Calculate overlays
            ema20 = ta.ema(df["Close"], length=20)
            ema50 = ta.ema(df["Close"], length=50)
            bb = ta.bbands(df["Close"], length=20, std=2)

            add_plots = []
            if ema20 is not None:
                add_plots.append(mpf.make_addplot(ema20, color="blue", width=0.8))
            if ema50 is not None:
                add_plots.append(mpf.make_addplot(ema50, color="orange", width=0.8))
            if bb is not None and len(bb.columns) >= 3:
                add_plots.append(mpf.make_addplot(bb.iloc[:, 0], color="gray", linestyle="--", width=0.5))
                add_plots.append(mpf.make_addplot(bb.iloc[:, 2], color="gray", linestyle="--", width=0.5))

            # Generate chart
            buf = io.BytesIO()
            mpf.plot(
                df,
                type="candle",
                style="charles",
                title=f"{ticker} — {len(df)} candles",
                volume=True,
                addplot=add_plots if add_plots else None,
                figsize=(14, 8),
                savefig=dict(fname=buf, dpi=150, bbox_inches="tight"),
            )
            buf.seek(0)
            image_bytes = buf.read()

            # Also save to disk
            chart_path = self.output_dir / f"{ticker}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.png"
            with open(chart_path, "wb") as f:
                f.write(image_bytes)

            logger.info("Chart generated for %s (%d candles, %d KB)", ticker, len(df), len(image_bytes) // 1024)
            return image_bytes

        except ImportError as e:
            logger.error("mplfinance not installed: %s", e)
            return None
        except Exception as e:
            logger.error("Chart generation failed for %s: %s", ticker, e)
            return None

    def analyze_chart(self, df: pd.DataFrame, ticker: str) -> dict | None:
        """Generate chart and analyze with Claude Vision. Returns structured analysis."""
        image_bytes = self.generate_chart_image(df, ticker)
        if not image_bytes:
            return None

        try:
            import anthropic
            from src.ai.prompts.sentiment import build_chart_analysis_prompt

            # Encode image to base64
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            # Need to call Claude directly (not through router — vision needs special handling)
            if not self.ai.claude:
                logger.warning("Claude not configured — skipping chart analysis")
                return None

            client = self.ai.claude.client
            model = self.ai.models.get("quality", "claude-sonnet-4-6")

            response = client.messages.create(
                model=model,
                max_tokens=800,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": build_chart_analysis_prompt(),
                        },
                    ],
                }],
            )

            # Track tokens
            self.ai.claude.total_input_tokens += response.usage.input_tokens
            self.ai.claude.total_output_tokens += response.usage.output_tokens

            # Parse JSON response
            import json
            text = response.content[0].text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)

            result = json.loads(text)
            result["ticker"] = ticker
            logger.info(
                "Chart analysis %s: %s (conf=%d%%, patterns=%s)",
                ticker, result.get("recommendation", "?"),
                result.get("confidence", 0),
                result.get("patterns", []),
            )
            return result

        except Exception as e:
            logger.error("Chart analysis failed for %s: %s", ticker, e)
            return None
