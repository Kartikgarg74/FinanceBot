"""Centralized ticker configuration — all scripts import from here.

50 Indian tickers across 10 sectors, 5 per sector for real market coverage.
Covers all major Nifty 50 constituents + key sector representatives.
"""

# ── INDIAN MARKET (NSE) — 5 tickers per sector, 10 sectors ────────

INDIA_TICKERS = {
    # Banking & Finance (~35% Nifty weight)
    "banking": [
        "HDFCBANK",     # HDFC Bank — largest private bank
        "ICICIBANK",    # ICICI Bank — 2nd largest private bank
        "SBIN",         # State Bank of India — largest PSU bank
        "KOTAKBANK",    # Kotak Mahindra Bank
        "AXISBANK",     # Axis Bank
    ],

    # NBFC & Financial Services
    "nbfc": [
        "BAJFINANCE",   # Bajaj Finance — NBFC leader
        "BAJAJFINSV",   # Bajaj Finserv — holding company
        "SBILIFE",      # SBI Life Insurance
        "HDFCLIFE",     # HDFC Life Insurance
        "INDUSINDBK",   # IndusInd Bank
    ],

    # IT & Technology (~15% Nifty weight)
    "it": [
        "TCS",          # Tata Consultancy Services
        "INFY",         # Infosys
        "HCLTECH",      # HCL Technologies
        "WIPRO",        # Wipro
        "TECHM",        # Tech Mahindra
    ],

    # Oil, Gas & Energy
    "energy": [
        "RELIANCE",     # Reliance Industries — largest by market cap
        "ONGC",         # Oil & Natural Gas Corp
        "NTPC",         # NTPC — power generation
        "POWERGRID",    # Power Grid Corp
        "BPCL",         # Bharat Petroleum
    ],

    # Consumer & FMCG
    "consumer": [
        "HINDUNILVR",   # Hindustan Unilever
        "ITC",          # ITC Limited
        "NESTLEIND",    # Nestle India
        "BRITANNIA",    # Britannia Industries
        "TATACONSUM",   # Tata Consumer Products
    ],

    # Automobile & Auto Components
    "auto": [
        "MARUTI",       # Maruti Suzuki — largest carmaker
        "M&M",          # Mahindra & Mahindra
        "BAJAJ-AUTO",   # Bajaj Auto — two-wheelers
        "EICHERMOT",    # Eicher Motors (Royal Enfield)
        "HEROMOTOCO",   # Hero MotoCorp
    ],

    # Metals, Mining & Materials
    "metals": [
        "TATASTEEL",    # Tata Steel
        "JSWSTEEL",     # JSW Steel
        "HINDALCO",     # Hindalco (Aluminium)
        "COALINDIA",    # Coal India
        "GRASIM",       # Grasim Industries (Aditya Birla)
    ],

    # Pharma & Healthcare
    "pharma": [
        "SUNPHARMA",    # Sun Pharma — largest pharma
        "DRREDDY",      # Dr. Reddy's Laboratories
        "CIPLA",        # Cipla
        "DIVISLAB",     # Divi's Laboratories
        "APOLLOHOSP",   # Apollo Hospitals
    ],

    # Telecom & Media
    "telecom": [
        "BHARTIARTL",   # Bharti Airtel — largest telecom
        "LT",           # Larsen & Toubro — infrastructure
        "ADANIENT",     # Adani Enterprises
        "ADANIPORTS",   # Adani Ports
        "TITAN",        # Titan Company (Tata - jewellery/watches)
    ],

    # Cement & Infrastructure
    "cement_infra": [
        "ULTRACEMCO",   # UltraTech Cement — largest cement
        "SHRIRAMFIN",   # Shriram Finance
        "ASIANPAINT",   # Asian Paints — market leader
        "LTIM",         # LTIMindtree (IT services)
        "TRENT",        # Trent (Tata - Westside retail)
    ],
}

# Flat list of all Indian tickers
INDIA_TICKERS_FLAT = []
for sector_tickers in INDIA_TICKERS.values():
    INDIA_TICKERS_FLAT.extend(sector_tickers)

# ── ALL MARKETS CONFIG ──────────────────────────────────────────────

TICKERS_CONFIG = {
    "india": {
        "tickers": INDIA_TICKERS_FLAT,
        "interval_daily": "1d",
        "interval_hourly": "1h",
        "days_daily": 365,
        "days_hourly": 59,
        "broker": "zerodha",
        "trade_type": "intraday",
        "sectors": INDIA_TICKERS,
    },
}

# ── Sector classification for cross-sector analysis ─────────────────

TICKER_TO_SECTOR = {}
for sector, tickers in INDIA_TICKERS.items():
    for ticker in tickers:
        TICKER_TO_SECTOR[ticker] = sector


def get_tickers(market: str = "india") -> list[str]:
    """Get all tickers for a market."""
    return TICKERS_CONFIG[market]["tickers"]


def get_sector(ticker: str) -> str:
    """Get sector for a ticker."""
    return TICKER_TO_SECTOR.get(ticker, "unknown")


def get_tickers_by_sector(sector: str) -> list[str]:
    """Get all tickers in a sector."""
    return INDIA_TICKERS.get(sector, [])


def get_all_sectors() -> list[str]:
    """Get all sector names."""
    return list(INDIA_TICKERS.keys())
