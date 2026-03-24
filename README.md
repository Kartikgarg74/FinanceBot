# FinanceBot

AI-powered trading bot that analyzes charts, scrapes financial news, and executes buy/sell decisions across three markets: Indian stocks (Zerodha), US stocks (Alpaca), and cryptocurrency (Binance).

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                    FinanceBot                           │
│                                                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │
│  │   Zerodha    │ │    Alpaca    │ │   Binance    │  │
│  │ Indian Stocks│ │  US Stocks   │ │    Crypto    │  │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘  │
│         │                │                │           │
│  ┌──────▼────────────────▼────────────────▼───────┐  │
│  │          Shared Analysis Engine                  │  │
│  │  ┌─────────────┐ ┌────────────┐ ┌────────────┐ │  │
│  │  │  Technical   │ │  News &    │ │   Chart    │ │  │
│  │  │  Analysis    │ │ Sentiment  │ │   Vision   │ │  │
│  │  │  (pandas-ta) │ │  (Claude)  │ │  (Claude)  │ │  │
│  │  └─────────────┘ └────────────┘ └────────────┘ │  │
│  └────────────────────────┬───────────────────────┘  │
│                           │                           │
│  ┌────────────────────────▼───────────────────────┐  │
│  │              Risk Management                    │  │
│  │  Position sizing · Stop-loss · Circuit breakers │  │
│  └────────────────────────┬───────────────────────┘  │
│                           │                           │
│  ┌──────────┐  ┌──────────▼──┐  ┌─────────────────┐ │
│  │ Telegram │  │   SQLite    │  │  AI Router      │ │
│  │   Bot    │  │   Database  │  │ Claude + Groq   │ │
│  └──────────┘  └─────────────┘  └─────────────────┘ │
└────────────────────────────────────────────────────────┘
```

## Workflow

### Trading Cycle (runs on schedule per market)

```
1. FETCH DATA
   ├── Live quotes (Kite WebSocket / Alpaca API / Binance WebSocket)
   ├── Financial news (Finnhub, Alpha Vantage, MarketAux, Reddit)
   └── Candlestick data for chart generation

2. TECHNICAL ANALYSIS
   ├── RSI, MACD, Bollinger Bands, EMA (20/50/200), VWAP, ATR
   ├── Multi-timeframe: 5m, 15m, 1h, 4h, daily
   └── Signal: BUY / SELL / HOLD

3. SENTIMENT ANALYSIS (Claude Haiku)
   ├── Score each news article: -5 (bearish) to +5 (bullish)
   ├── Weighted average per ticker (recency × confidence × source reliability)
   └── Overlay on technical signals

4. CHART VISION ANALYSIS (Claude Sonnet — 2-4x/day per ticker)
   ├── Generate candlestick chart with indicators (matplotlib/mplfinance)
   ├── Send to Claude Vision API for pattern recognition
   └── Identify: trends, support/resistance, chart patterns, volume signals

5. SIGNAL GENERATION
   ├── Combine: technical + sentiment + chart vision
   ├── Output: BUY/SELL/HOLD + confidence (0-100%)
   └── Only act on confidence > 70%

6. RISK MANAGEMENT
   ├── Position sizing (2% risk per trade)
   ├── Check: daily loss limit, max positions, max drawdown
   ├── Stop-loss + take-profit calculation
   └── Circuit breaker: pause after 5 consecutive losses

7. ORDER EXECUTION
   ├── Place order via broker API
   ├── Log trade to SQLite with full reasoning
   └── Telegram notification with entry details

8. MONITORING (24/7)
   ├── Daily P&L summary at market close
   ├── Risk warnings at 80% of limits
   └── Telegram commands: /status, /trades, /pause, /resume
```

### Market-Specific Schedules

| Market | Hours | Scan Interval | Chart Analysis |
|--------|-------|---------------|----------------|
| Zerodha (NSE/BSE) | 9:15 AM - 3:30 PM IST | Every 15 min | Every 6 hours |
| Alpaca (NYSE/NASDAQ) | 9:30 AM - 4:00 PM ET | Every 15 min | Every 6 hours |
| Binance (Crypto) | 24/7 | Every 5 min | Every 6 hours |

## APIs Used

### Broker APIs (Order Execution)

| API | Purpose | Cost | Rate Limit |
|-----|---------|------|------------|
| **Zerodha Kite Connect** | Indian stock trading (NSE/BSE). Orders, portfolio, holdings, P&L. | Free for personal use (since March 2025) | Standard API limits |
| **Alpaca Markets API** | US stock trading. Commission-free. Paper + live trading. | Free | 200 req/min |
| **Binance API** (via CCXT) | Crypto spot trading. BTC, ETH, SOL, BNB, XRP pairs. | Free | 6,000 req/min |

### Data APIs (News & Market Data)

| API | Purpose | Cost | Rate Limit |
|-----|---------|------|------------|
| **Finnhub** | Global stock & crypto news, real-time quotes | Free tier | 60 calls/min |
| **Alpha Vantage** | News with built-in AI sentiment scores | Free tier | 25 calls/day |
| **MarketAux** | Stock + crypto news with sentiment | Free tier | 100 calls/day |
| **Reddit (PRAW)** | Sentiment from r/IndianStreetBets, r/wallstreetbets, r/cryptocurrency | Free | 60 calls/min |
| **yfinance** | Historical price data (5+ years) for backtesting | Free | Unofficial, no hard limit |

### AI APIs (Analysis & Decisions)

| API | Purpose | Model | Cost |
|-----|---------|-------|------|
| **Anthropic Claude API** | News sentiment scoring | Claude Haiku 4.5 | ~$0.01/article |
| **Anthropic Claude API** | Chart pattern recognition (Vision) | Claude Sonnet 4.6 | ~$0.05-0.10/chart |
| **Anthropic Claude API** | Trade signal reasoning | Claude Haiku 4.5 | ~$0.01/signal |
| **Groq** (fallback) | All tasks when Claude is unavailable | Llama 3.3 70B | Free (14,400 req/day) |

### MCP Integrations

| MCP Server | Purpose | Capabilities |
|------------|---------|-------------|
| `zerodha/kite-mcp-server` | Zerodha integration for Claude Desktop | Portfolio, holdings, quotes, P&L (read-only) |
| `nirholas/Binance-MCP` | Full Binance integration (478+ tools) | Spot trading, wallet, staking, market data |
| `atilaahmettaner/tradingview-mcp` | TradingView screening & alerts | Screening, technical analysis, alerts |

### Notifications

| Service | Purpose |
|---------|---------|
| **Telegram Bot API** | Trade alerts, daily summaries, commands (/status, /trades, /pause, /risk) |
| **Email (SMTP/Gmail)** | Optional weekly reports |

## Tech Stack

- **Language:** Python 3.11+
- **Trading:** kiteconnect, alpaca-py, ccxt, freqtrade
- **Technical Analysis:** pandas-ta, mplfinance, backtrader
- **AI:** anthropic (Claude API), groq
- **Data:** yfinance, finnhub-python, praw
- **Database:** SQLAlchemy + SQLite
- **Scheduling:** APScheduler
- **Notifications:** python-telegram-bot
- **Real-time:** websockets (Binance streaming)
- **Security:** cryptography (Fernet encryption for secrets at rest)
- **Deployment:** Docker + docker-compose

## Setup

```bash
# Clone
git clone https://github.com/Kartikgarg74/FinanceBot.git
cd FinanceBot

# Virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run (paper trading mode by default)
python main.py --market zerodha     # Indian stocks only
python main.py --market alpaca      # US stocks only
python main.py --market binance     # Crypto only
python main.py --market all         # All markets
python main.py --scan-once          # Single scan cycle (testing)
python main.py --mode live          # Live trading (use with caution)
```

### Docker

```bash
docker-compose up -d
```

## Risk Management

- **Paper trading by default** — must explicitly switch to live
- **2% max risk per trade** — position sizing based on stop-loss distance
- **5% max daily loss** — all trading pauses if hit
- **15% max drawdown** — emergency shutdown from portfolio peak
- **Circuit breaker** — pause after 5 consecutive losses for 24 hours
- **SEBI compliant** — all Indian stock orders via official Kite Connect API

## Estimated Monthly Cost

| Component | Cost |
|-----------|------|
| Claude API (sentiment + charts) | ~$10-15 |
| Groq (fallback) | Free |
| All news APIs | Free |
| All broker APIs | Free |
| **Total** | **~$10-15/month** |

## Telegram Commands

```
/status          - Portfolio status across all markets
/zerodha         - Indian stock positions & P&L
/alpaca          - US stock positions & P&L
/crypto          - Crypto positions & P&L
/trades          - Recent trade log
/signals         - Current active signals
/pause [module]  - Pause trading
/resume [module] - Resume trading
/risk            - Current risk metrics
/news            - Latest scored news
/backtest        - Run backtest
```

## License

Personal use. Not financial advice. Trade at your own risk.
