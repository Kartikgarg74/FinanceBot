"""Zerodha Kite Connect client with automated TOTP login and order execution."""

import logging
import time
from datetime import datetime
from typing import Optional

import pandas as pd

from ..base_trader import BaseTrader, OrderSide, OrderType, TradeRecord, Position

logger = logging.getLogger(__name__)


class ZerodhaClient(BaseTrader):
    """
    Zerodha Kite Connect wrapper.
    - Free personal API (no market data, but orders work)
    - Historical data via yfinance (free)
    - Auto TOTP login for daily token refresh
    """

    def __init__(self, config: dict, mode: str = "paper"):
        super().__init__(config, mode)
        self.zerodha_config = config.get("zerodha", {})
        self.trading_config = config.get("trading", {})
        self.kite = None
        self._access_token = None

    def connect(self) -> bool:
        """Connect to Zerodha. Handles TOTP-based auto login."""
        if self.mode == "paper":
            logger.info("[PAPER] Zerodha client in paper mode — no real connection")
            self._paper_capital = float(self.trading_config.get("max_capital_per_trade", 50000)) * 10
            return True

        api_key = self.zerodha_config.get("api_key", "")
        api_secret = self.zerodha_config.get("api_secret", "")

        if not api_key or not api_secret:
            logger.error("Zerodha API key/secret not configured")
            return False

        try:
            from kiteconnect import KiteConnect
            self.kite = KiteConnect(api_key=api_key)

            # Try auto-login with TOTP
            access_token = self._auto_login()
            if access_token:
                self.kite.set_access_token(access_token)
                self._access_token = access_token
                logger.info("Zerodha connected successfully (auto-login)")
                return True
            else:
                # Manual login required
                login_url = self.kite.login_url()
                logger.warning("Auto-login failed. Manual login required: %s", login_url)
                return False

        except Exception as e:
            logger.error("Zerodha connection failed: %s", e)
            return False

    def _auto_login(self) -> Optional[str]:
        """Automate Zerodha login using TOTP. All errors sanitized to prevent secret leakage."""
        from src.utils.security import sanitize_error

        totp_secret = self.zerodha_config.get("totp_secret", "")
        user_id = self.zerodha_config.get("user_id", "")
        password = self.zerodha_config.get("password", "")
        api_key = self.zerodha_config.get("api_key", "")
        api_secret = self.zerodha_config.get("api_secret", "")

        if not all([totp_secret, user_id, password]):
            logger.info("TOTP auto-login not configured (missing credentials)")
            return None

        try:
            import pyotp
            import httpx

            totp = pyotp.TOTP(totp_secret)

            with httpx.Client(follow_redirects=True) as client:
                # Step 1: Login with user_id and password
                login_resp = client.post(
                    "https://kite.zerodha.com/api/login",
                    data={"user_id": user_id, "password": password},
                )
                login_data = login_resp.json()

                if login_data.get("status") != "success":
                    # Only log status, NOT the full response (may contain tokens)
                    logger.error("Zerodha login step 1 failed (status=%s)", login_data.get("status"))
                    return None

                request_id = login_data["data"]["request_id"]

                # Step 2: 2FA with TOTP
                twofa_resp = client.post(
                    "https://kite.zerodha.com/api/twofa",
                    data={
                        "user_id": user_id,
                        "request_id": request_id,
                        "twofa_value": totp.now(),
                        "twofa_type": "totp",
                    },
                )
                twofa_data = twofa_resp.json()

                if twofa_data.get("status") != "success":
                    logger.error("Zerodha 2FA failed (status=%s)", twofa_data.get("status"))
                    return None

                # Step 3: Get request_token via Kite login URL
                # Note: api_key is passed as URL param (required by Zerodha's flow)
                login_url = f"https://kite.trade/connect/login?api_key={api_key}&v=3"
                resp = client.get(login_url)

                # Extract request_token from redirect URL
                final_url = str(resp.url)
                if "request_token=" in final_url:
                    request_token = final_url.split("request_token=")[1].split("&")[0]
                else:
                    # Don't log the full URL (contains api_key)
                    logger.error("Could not extract request_token from Kite redirect")
                    return None

                # Step 4: Exchange for access_token
                from kiteconnect import KiteConnect
                kite = KiteConnect(api_key=api_key)
                session = kite.generate_session(request_token, api_secret=api_secret)
                access_token = session["access_token"]

                logger.info("Zerodha auto-login successful for user %s", user_id[:3] + "***")
                return access_token

        except Exception as e:
            logger.error("Zerodha auto-login error: %s", sanitize_error(e))
            return None

    def get_positions(self) -> list[Position]:
        if self.mode == "paper":
            return list(self._paper_portfolio.values())

        try:
            positions_data = self.kite.positions()
            positions = []
            for p in positions_data.get("net", []):
                qty = p.get("quantity", 0)
                if qty == 0:
                    continue
                avg = p.get("average_price", 0)
                ltp = p.get("last_price", 0)
                pnl = (ltp - avg) * qty
                positions.append(Position(
                    ticker=p["tradingsymbol"],
                    quantity=qty,
                    avg_price=avg,
                    current_price=ltp,
                    pnl=pnl,
                    pnl_pct=(pnl / (avg * qty) * 100) if avg * qty > 0 else 0,
                    exchange="ZERODHA",
                ))
            return positions
        except Exception as e:
            logger.error("Failed to get Zerodha positions: %s", e)
            return []

    def get_balance(self) -> float:
        if self.mode == "paper":
            return self._paper_capital

        try:
            margins = self.kite.margins()
            return margins.get("equity", {}).get("available", {}).get("live_balance", 0)
        except Exception as e:
            logger.error("Failed to get Zerodha balance: %s", e)
            return 0

    def place_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: int | float,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> TradeRecord:
        if self.mode == "paper":
            from ..base_trader import Signal, SignalAction
            signal = Signal(
                ticker=ticker,
                action=SignalAction.BUY if side == OrderSide.BUY else SignalAction.SELL,
                confidence=100,
                price=price or 0,
            )
            return self._paper_trade(signal, quantity)

        from kiteconnect import KiteConnect

        order_type_map = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP_LOSS: "SL",
            OrderType.STOP_LOSS_MARKET: "SL-M",
        }

        product = self.trading_config.get("product_types", ["CNC"])[0]
        exchange = self.trading_config.get("allowed_exchanges", ["NSE"])[0]

        try:
            params = {
                "tradingsymbol": ticker,
                "exchange": exchange,
                "transaction_type": side.value,
                "quantity": int(quantity),
                "order_type": order_type_map.get(order_type, "MARKET"),
                "product": product,
                "variety": "regular",
            }
            if price and order_type != OrderType.MARKET:
                params["price"] = price
            if stop_loss:
                params["trigger_price"] = stop_loss

            order_id = self.kite.place_order(**params)

            return TradeRecord(
                id=str(order_id),
                ticker=ticker,
                side=side.value,
                quantity=int(quantity),
                price=price or 0,
                order_type=order_type_map.get(order_type, "MARKET"),
                status="executed",
                exchange="ZERODHA",
            )
        except Exception as e:
            logger.error("Zerodha order failed: %s", e)
            return TradeRecord(
                id="failed",
                ticker=ticker,
                side=side.value,
                quantity=int(quantity),
                price=price or 0,
                order_type=order_type_map.get(order_type, "MARKET"),
                status="failed",
                reason=str(e),
                exchange="ZERODHA",
            )

    def get_quote(self, ticker: str) -> dict:
        if self.mode == "paper":
            # Use yfinance for paper mode quotes
            from ..analysis.data_fetcher import DataFetcher
            df = DataFetcher().fetch_indian_stock(ticker, interval="1d", days=5)
            if df is not None and not df.empty:
                row = df.iloc[-1]
                return {
                    "ticker": ticker,
                    "price": float(row["Close"]),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "volume": int(row["Volume"]) if "Volume" in row else 0,
                }
            return {"ticker": ticker, "price": 0}

        try:
            exchange = self.trading_config.get("allowed_exchanges", ["NSE"])[0]
            data = self.kite.quote(f"{exchange}:{ticker}")
            q = list(data.values())[0] if data else {}
            return {
                "ticker": ticker,
                "price": q.get("last_price", 0),
                "open": q.get("ohlc", {}).get("open", 0),
                "high": q.get("ohlc", {}).get("high", 0),
                "low": q.get("ohlc", {}).get("low", 0),
                "volume": q.get("volume", 0),
            }
        except Exception as e:
            logger.error("Quote failed for %s: %s", ticker, e)
            return {"ticker": ticker, "price": 0}

    def get_historical_data(self, ticker: str, interval: str = "1d", days: int = 365):
        """Get historical data via yfinance (FREE)."""
        from ..analysis.data_fetcher import DataFetcher
        return DataFetcher().fetch_indian_stock(ticker, interval, days)

    def get_holdings(self) -> list[dict]:
        """Get long-term holdings (CNC positions)."""
        if self.mode == "paper":
            return [{"ticker": p.ticker, "quantity": p.quantity, "avg_price": p.avg_price}
                    for p in self._paper_portfolio.values()]
        try:
            return self.kite.holdings()
        except Exception as e:
            logger.error("Failed to get holdings: %s", e)
            return []
