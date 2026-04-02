"""Financial performance metrics with frequency-aware annualization.

Fixes the common error of using sqrt(252) for all data frequencies.
"""

import numpy as np

# Approximate trading periods per year by interval
PERIODS_PER_YEAR = {
    "1m": 252 * 390,       # 390 minutes per US trading day
    "5m": 252 * 78,        # 78 five-minute bars per day
    "15m": 252 * 26,       # 26 fifteen-minute bars per day
    "30m": 252 * 13,
    "1h": 252 * 6.5,       # 6.5 hours per US trading day
    "4h": 252 * 1.625,
    "1d": 252,
    "1wk": 52,
    "1mo": 12,
}


def get_periods_per_year(interval: str = "1d") -> float:
    """Get annualization factor for a given bar interval."""
    return PERIODS_PER_YEAR.get(interval, 252)


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0,
                 interval: str = "1d") -> float:
    """Annualized Sharpe ratio with frequency-aware scaling."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    periods = get_periods_per_year(interval)
    excess = returns - risk_free_rate / periods
    return float(excess.mean() / excess.std() * np.sqrt(periods))


def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0,
                  interval: str = "1d") -> float:
    """Annualized Sortino ratio (only penalizes downside volatility)."""
    if len(returns) < 2:
        return 0.0
    periods = get_periods_per_year(interval)
    excess = returns - risk_free_rate / periods
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(periods))


def calmar_ratio(returns: np.ndarray, interval: str = "1d") -> float:
    """Annualized return / max drawdown."""
    if len(returns) < 2:
        return 0.0
    periods = get_periods_per_year(interval)
    annual_return = returns.mean() * periods

    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    max_dd = abs(drawdown.min())

    return float(annual_return / max_dd) if max_dd > 0 else 0.0


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown as a negative percentage."""
    if len(returns) == 0:
        return 0.0
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return float(drawdown.min())


def profit_factor(returns: np.ndarray) -> float:
    """Gross profits / gross losses."""
    wins = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return float(wins / losses) if losses > 0 else float("inf")


def compute_all_metrics(returns: np.ndarray, interval: str = "1d",
                        risk_free_rate: float = 0.0) -> dict:
    """Compute all performance metrics at once."""
    return {
        "sharpe": sharpe_ratio(returns, risk_free_rate, interval),
        "sortino": sortino_ratio(returns, risk_free_rate, interval),
        "calmar": calmar_ratio(returns, interval),
        "max_drawdown": max_drawdown(returns),
        "profit_factor": profit_factor(returns),
        "win_rate": float((returns > 0).mean()) if len(returns) > 0 else 0,
        "mean_return": float(returns.mean()) if len(returns) > 0 else 0,
        "total_return": float(np.prod(1 + returns) - 1) if len(returns) > 0 else 0,
        "n_trades": int((returns != 0).sum()),
        "interval": interval,
        "annualization_factor": get_periods_per_year(interval),
    }
