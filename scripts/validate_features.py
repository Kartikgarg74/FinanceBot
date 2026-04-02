#!/usr/bin/env python3
"""
Feature & Pattern Validation — Are the learned patterns real or noise?

This script answers:
1. Which features appear consistently across ALL tickers? (Cross-ticker stability)
2. Do features appear consistently across walk-forward folds? (Temporal stability)
3. Do the top features make economic sense? (Domain validation)
4. Is the model relying on spurious correlations? (Noise detection)
5. Feature predictive power tests (individual feature significance)
6. Overfitting indicators (CV vs holdout gap, Sharpe inflation)

Usage:
    python scripts/validate_features.py
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.finance.analysis.data_fetcher import DataFetcher
from src.ml.feature_pipeline import FeaturePipeline
from src.ml.label_generator import LabelGenerator
from src.finance.cost_model import estimate_round_trip_pct

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("validate_features")

MODEL_DIR = Path("data/ml_models")

# ── Economic meaning of each feature ───────────────────────────────
FEATURE_ECONOMICS = {
    # TREND — momentum/mean-reversion indicators
    "ema_ratio_8_21": ("TREND", "Short vs medium-term trend alignment. >1 = bullish, <1 = bearish. "
                       "Well-established in trend-following literature."),
    "ema_ratio_21_50": ("TREND", "Medium vs long-term trend. Captures regime changes. "
                        "Validated by Jegadeesh & Titman (1993) momentum factor."),
    "price_to_ema21": ("TREND", "Distance from 21-day mean. Mean-reversion signal. "
                       "Supported by Poterba & Summers (1988)."),
    "price_to_ema50": ("TREND", "Distance from 50-day mean. Longer-term mean-reversion. Valid."),

    # MOMENTUM
    "rsi_14": ("MOMENTUM", "Relative Strength Index. Overbought/oversold. "
               "Widely used but mixed academic evidence as standalone."),
    "rsi_14_slope": ("MOMENTUM", "Rate of change of RSI. Captures momentum acceleration. "
                     "Novel feature — needs careful validation."),
    "macd_histogram": ("MOMENTUM", "MACD momentum. Trend-following signal. "
                       "Appel (1979). Common in practice, moderate academic support."),
    "macd_hist_slope": ("MOMENTUM", "MACD momentum acceleration. 2nd derivative of price. "
                        "Captures inflection points. Economically meaningful."),
    "roc_5": ("MOMENTUM", "5-bar rate of change. Short-term momentum. Valid."),
    "roc_10": ("MOMENTUM", "10-bar rate of change. Medium-term momentum. Valid."),
    "stoch_k": ("MOMENTUM", "Stochastic oscillator. Mean-reversion in bounded range. "
                "Lane (1984). Moderate evidence."),

    # VOLATILITY
    "atr_pct": ("VOLATILITY", "Normalized ATR. Volatility regime indicator. "
                "High ATR → trending market, low ATR → range-bound. Economically sound."),
    "bb_pct_b": ("VOLATILITY", "Bollinger Band %B. Position within volatility bands. "
                 "Bollinger (2002). Moderate evidence for mean-reversion."),
    "bb_width_pct": ("VOLATILITY", "Bollinger Band width. Volatility squeeze detection. "
                     "Precedes breakouts. Economically valid."),
    "vol_10": ("VOLATILITY", "10-bar realized volatility. Short-term risk. "
               "Core factor in volatility clustering (Mandelbrot, 1963)."),
    "vol_20": ("VOLATILITY", "20-bar realized volatility. Medium-term risk. Valid."),
    "vol_ratio": ("VOLATILITY", "Short/long volatility ratio. Detects volatility regime shifts. "
                  "Novel but economically motivated (volatility mean-reversion)."),

    # VOLUME
    "volume_ratio": ("VOLUME", "Current volume vs 20-day average. Detects unusual activity. "
                     "Volume-price confirmation is a core technical principle."),
    "obv_slope": ("VOLUME", "On-Balance Volume trend. Accumulation/distribution. "
                  "Granville (1963). Shows institutional buying/selling."),
    "mfi_14": ("VOLUME", "Money Flow Index. Volume-weighted RSI. "
               "Combines price and volume momentum. Moderate evidence."),
    "vol_price_divergence": ("VOLUME", "Price-volume divergence. Price moves WITHOUT volume confirmation. "
                            "Classic warning signal. Strong economic basis."),

    # PRICE ACTION
    "body_ratio": ("PRICE_ACTION", "Candle body as fraction of range. "
                   "Large body = conviction, small = indecision. "
                   "Japanese candlestick theory (Nison, 1991)."),
    "upper_shadow": ("PRICE_ACTION", "Upper wick length. Selling pressure at highs. "
                     "Rejection pattern. Candlestick theory."),
    "lower_shadow": ("PRICE_ACTION", "Lower wick length. Buying pressure at lows. "
                     "Support/demand pattern. Candlestick theory."),
    "gap_pct": ("PRICE_ACTION", "Overnight gap size. Reflects overnight news/sentiment. "
                "Gap fill patterns well-documented in intraday trading."),
    "pct_from_20d_high": ("PRICE_ACTION", "Distance from recent high. Resistance proximity. "
                          "Support/resistance is well-established."),
    "pct_from_20d_low": ("PRICE_ACTION", "Distance from recent low. Support proximity. Valid."),

    # STATISTICAL
    "z_score_20": ("STATISTICAL", "Z-score of price. Mean-reversion indicator. "
                   "Statistical arbitrage basis. Strong theoretical support."),
    "skew_20": ("STATISTICAL", "Return distribution skewness. Tail risk indicator. "
                "Negative skew = crash risk premium. Harvey & Siddique (2000)."),
    "kurt_20": ("STATISTICAL", "Return distribution kurtosis. Fat tail frequency. "
                "Higher kurtosis = more extreme moves. Mandelbrot (1963)."),

    # LAG
    "return_lag_1": ("LAG", "Yesterday's return. Short-term autocorrelation/reversal. "
                     "1-day reversal is well-documented (Lo & MacKinlay, 1990). "
                     "WARNING: Strong lag-1 reliance may indicate data leakage."),
    "return_lag_2": ("LAG", "2-day lagged return. Momentum/reversal. "
                     "Should be weaker than lag_1. If dominant → possible overfitting."),
    "return_lag_3": ("LAG", "3-day lagged return. Weak momentum signal. "
                     "Legitimate but noisy. High importance is suspicious."),

    # TIME
    "day_of_week": ("TIME", "Day-of-week effect. Monday/Friday anomalies. "
                    "French (1980) documented Monday effect. "
                    "WARNING: Largely arbitraged away in modern markets. "
                    "High importance suggests overfitting to calendar patterns."),
}

# Features that are RED FLAGS if they're #1 or dominant
SUSPICIOUS_IF_DOMINANT = {
    "day_of_week": "Calendar effects are largely arbitraged away. Model may be overfitting to day patterns.",
    "return_lag_1": "Strong lag-1 dependence may indicate information leakage or naive autocorrelation.",
    "return_lag_2": "Lagged returns as top feature suggests model is just following recent momentum.",
    "return_lag_3": "3-day lag as dominant feature is unusual — verify no look-ahead bias.",
    "kurt_20": "Kurtosis as #1 feature is unusual — may be fitting to extreme events.",
}


def load_all_metadata() -> list[dict]:
    """Load metadata for all trained models."""
    results = []
    for p in sorted(MODEL_DIR.glob("*_metadata.json")):
        meta = json.loads(p.read_text())
        results.append(meta)
    return results


def analyze_cross_ticker_features(all_meta: list[dict]):
    """Which features are consistently important across ALL tickers?"""
    print("\n" + "=" * 80)
    print("1. CROSS-TICKER FEATURE CONSISTENCY")
    print("   (Do the same features matter for different stocks?)")
    print("=" * 80)

    # Count how many tickers each feature appears in top-10
    top10_counts = Counter()
    top5_counts = Counter()
    top1_features = []

    for meta in all_meta:
        top_feats = meta.get("top_features", [])
        if isinstance(top_feats, list) and top_feats:
            for i, item in enumerate(top_feats[:10]):
                name = item[0] if isinstance(item, (list, tuple)) else item
                top10_counts[name] += 1
                if i < 5:
                    top5_counts[name] += 1
                if i == 0:
                    top1_features.append((meta["ticker"], name))

    n_tickers = len(all_meta)
    print(f"\nAnalyzed {n_tickers} tickers.\n")

    print("Features appearing in TOP-5 across tickers:")
    print(f"{'Feature':<30} {'Count':>6} {'% Tickers':>10} {'Verdict'}")
    print("-" * 75)
    for feat, count in top5_counts.most_common(15):
        pct = count / n_tickers * 100
        if pct >= 60:
            verdict = "STRONG — consistent signal"
        elif pct >= 40:
            verdict = "MODERATE — appears often"
        elif pct >= 20:
            verdict = "WEAK — ticker-specific"
        else:
            verdict = "RARE — likely noise"
        print(f"  {feat:<28} {count:>6} {pct:>9.0f}% {verdict}")

    # Universal features (appear in top-10 for 70%+ tickers)
    universal = [f for f, c in top10_counts.items() if c >= n_tickers * 0.7]
    ticker_specific = [f for f, c in top10_counts.items() if c <= max(1, n_tickers * 0.2)]

    print(f"\nUniversal features (>70% of tickers): {universal if universal else 'NONE'}")
    print(f"Ticker-specific features (<20%): {len(ticker_specific)} features")

    print(f"\n#1 feature per ticker:")
    for ticker, feat in top1_features:
        warning = " ⚠ SUSPICIOUS" if feat in SUSPICIOUS_IF_DOMINANT else ""
        print(f"  {ticker:<12} → {feat}{warning}")

    return top5_counts, top10_counts


def analyze_temporal_stability(all_meta: list[dict]):
    """Are features stable across walk-forward folds?"""
    print("\n" + "=" * 80)
    print("2. TEMPORAL STABILITY (Walk-Forward Fold Consistency)")
    print("   (Do the same features matter in different time periods?)")
    print("=" * 80)

    stable_count = 0
    for meta in all_meta:
        stab = meta.get("feature_stability", {})
        jaccard = stab.get("mean_jaccard_similarity", 0)
        common = stab.get("common_top_features", [])
        is_stable = stab.get("stable", False)

        status = "STABLE" if is_stable else "UNSTABLE"
        if stable_count == 0 and not is_stable:
            pass
        if is_stable:
            stable_count += 1

        print(f"  {meta['ticker']:<12} Jaccard={jaccard:.3f} [{status}] "
              f"Common features: {common if common else 'NONE'}")

    pct_stable = stable_count / len(all_meta) * 100
    print(f"\nStable tickers: {stable_count}/{len(all_meta)} ({pct_stable:.0f}%)")

    if pct_stable < 50:
        print("\n  ⚠ WARNING: Most models have UNSTABLE features across folds.")
        print("  This means the model learns different patterns in different time periods.")
        print("  This is EXPECTED with limited daily data (246-500 bars).")
        print("  Mitigation: Use higher-frequency data (15-min) for more samples per fold.")
    else:
        print("\n  ✓ GOOD: Feature importance is relatively stable across time periods.")


def analyze_economic_validity(top5_counts: Counter):
    """Do the top features make economic sense?"""
    print("\n" + "=" * 80)
    print("3. ECONOMIC VALIDITY — Do the patterns make sense?")
    print("=" * 80)

    print(f"\n{'Feature':<30} {'Category':<15} {'Economic Justification'}")
    print("-" * 100)

    valid_count = 0
    suspicious_count = 0

    for feat, count in top5_counts.most_common(20):
        econ = FEATURE_ECONOMICS.get(feat, ("UNKNOWN", "Not documented — may be noise"))
        category, justification = econ

        is_suspicious = feat in SUSPICIOUS_IF_DOMINANT
        marker = "⚠" if is_suspicious else "✓"
        if is_suspicious:
            suspicious_count += 1
        else:
            valid_count += 1

        # Truncate justification for display
        just_short = justification[:60] + "..." if len(justification) > 60 else justification
        print(f"  {marker} {feat:<28} {category:<15} {just_short}")

    print(f"\nEconomically valid features: {valid_count}")
    print(f"Suspicious features: {suspicious_count}")


def analyze_overfitting_indicators(all_meta: list[dict]):
    """Check for signs of overfitting."""
    print("\n" + "=" * 80)
    print("4. OVERFITTING ANALYSIS")
    print("=" * 80)

    print(f"\n{'Ticker':<12} {'CV F1':>8} {'Hold F1':>9} {'Gap':>8} {'Gap%':>8} {'Verdict'}")
    print("-" * 60)

    for meta in all_meta:
        cv_f1 = meta.get("cv_f1", 0)
        hold_f1 = meta.get("holdout_f1", 0)
        gap = cv_f1 - hold_f1
        gap_pct = (gap / cv_f1 * 100) if cv_f1 > 0 else 0

        if gap_pct > 50:
            verdict = "SEVERE OVERFIT"
        elif gap_pct > 30:
            verdict = "MODERATE OVERFIT"
        elif gap_pct > 15:
            verdict = "MILD OVERFIT"
        elif gap_pct < 0:
            verdict = "HOLDOUT > CV (!)"
        else:
            verdict = "OK"

        print(f"  {meta['ticker']:<10} {cv_f1:>7.3f} {hold_f1:>8.3f} {gap:>+7.3f} {gap_pct:>7.1f}% {verdict}")

    avg_gap = np.mean([m.get("cv_f1", 0) - m.get("holdout_f1", 0) for m in all_meta])
    print(f"\nAverage CV→Holdout gap: {avg_gap:+.3f}")

    if avg_gap > 0.15:
        print("  ⚠ HIGH average gap — models are likely overfitting.")
        print("  Recommended: more data (higher frequency), stronger regularization,")
        print("  fewer features, or simpler models.")
    elif avg_gap > 0.05:
        print("  ⚠ MODERATE gap — some overfitting expected with limited data.")
    else:
        print("  ✓ Gap is small — models generalize reasonably well.")


def test_individual_feature_significance(ticker: str = "AAPL", market: str = "us"):
    """Test if individual top features have statistically significant predictive power."""
    print("\n" + "=" * 80)
    print(f"5. INDIVIDUAL FEATURE SIGNIFICANCE TEST — {ticker}")
    print("   (Can each feature alone predict direction better than random?)")
    print("=" * 80)

    from scipy.stats import spearmanr, pearsonr

    fetcher = DataFetcher()
    if market == "india":
        df = fetcher.fetch_indian_stock(ticker, "1d", 365)
    else:
        df = fetcher.fetch_us_stock(ticker, "1d", 730)

    if df is None or len(df) < 100:
        print(f"  Insufficient data for {ticker}")
        return

    pipeline = FeaturePipeline()
    features = pipeline.transform(df)
    features = pipeline.remove_correlated(features)

    # Future 1-bar return as target
    close = df["Close"]
    future_return = close.shift(-1) / close - 1
    common = features.index.intersection(future_return.dropna().index)
    X = features.loc[common]
    y = future_return.loc[common]

    print(f"\nSamples: {len(X)}, Features: {len(X.columns)}")
    print(f"\n{'Feature':<30} {'Spearman r':>12} {'p-value':>10} {'Pearson r':>12} {'p-value':>10} {'Verdict'}")
    print("-" * 90)

    sig_features = []
    for feat in X.columns:
        vals = X[feat].values
        target = y.values

        # Remove NaN pairs
        mask = ~(np.isnan(vals) | np.isnan(target) | np.isinf(vals))
        if mask.sum() < 30:
            continue

        sp_r, sp_p = spearmanr(vals[mask], target[mask])
        pe_r, pe_p = pearsonr(vals[mask], target[mask])

        if sp_p < 0.05 or pe_p < 0.05:
            verdict = "SIGNIFICANT"
            sig_features.append(feat)
        elif sp_p < 0.10 or pe_p < 0.10:
            verdict = "MARGINAL"
        else:
            verdict = "NOT SIG"

        # Only print top features or significant ones
        meta_path = MODEL_DIR / f"{ticker}_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            top_feat_names = [f[0] if isinstance(f, (list, tuple)) else f
                              for f in meta.get("top_features", [])[:15]]
        else:
            top_feat_names = []

        if feat in top_feat_names or verdict != "NOT SIG":
            marker = "★" if feat in top_feat_names[:5] else " "
            print(f"{marker} {feat:<28} {sp_r:>+11.4f} {sp_p:>9.4f} "
                  f"{pe_r:>+11.4f} {pe_p:>9.4f} {verdict}")

    print(f"\nSignificant features (p<0.05): {len(sig_features)}/{len(X.columns)}")
    print(f"Expected by chance (5%): {int(len(X.columns) * 0.05)}")

    if len(sig_features) <= int(len(X.columns) * 0.05) + 1:
        print("  ⚠ WARNING: Number of significant features is close to chance level.")
        print("  The model may be fitting noise rather than real patterns.")
    else:
        print("  ✓ More features significant than expected by chance alone.")


def generate_final_verdict(all_meta: list[dict], top5_counts: Counter):
    """Generate overall verdict on feature quality."""
    print("\n" + "=" * 80)
    print("6. FINAL VERDICT — Are the Patterns Real?")
    print("=" * 80)

    checks = []

    # Check 1: Cross-ticker consistency
    universal = [f for f, c in top5_counts.items() if c >= len(all_meta) * 0.5]
    if len(universal) >= 3:
        checks.append(("PASS", f"{len(universal)} features consistent across 50%+ tickers"))
    else:
        checks.append(("WARN", f"Only {len(universal)} features consistent — patterns may be ticker-specific"))

    # Check 2: Temporal stability
    stable = sum(1 for m in all_meta if m.get("feature_stability", {}).get("stable", False))
    if stable >= len(all_meta) * 0.5:
        checks.append(("PASS", f"{stable}/{len(all_meta)} tickers have stable features across time"))
    else:
        checks.append(("FAIL", f"Only {stable}/{len(all_meta)} tickers stable — features shift over time"))

    # Check 3: Overfitting gap
    avg_gap = np.mean([m.get("cv_f1", 0) - m.get("holdout_f1", 0) for m in all_meta])
    if avg_gap < 0.10:
        checks.append(("PASS", f"CV→Holdout gap is {avg_gap:.3f} — generalization OK"))
    elif avg_gap < 0.20:
        checks.append(("WARN", f"CV→Holdout gap is {avg_gap:.3f} — moderate overfitting"))
    else:
        checks.append(("FAIL", f"CV→Holdout gap is {avg_gap:.3f} — severe overfitting"))

    # Check 4: Suspicious features
    top3_per_ticker = []
    for meta in all_meta:
        top = meta.get("top_features", [])[:3]
        for item in top:
            name = item[0] if isinstance(item, (list, tuple)) else item
            top3_per_ticker.append(name)

    suspicious_pct = sum(1 for f in top3_per_ticker if f in SUSPICIOUS_IF_DOMINANT) / max(len(top3_per_ticker), 1) * 100
    if suspicious_pct < 20:
        checks.append(("PASS", f"Only {suspicious_pct:.0f}% of top-3 features are suspicious"))
    else:
        checks.append(("WARN", f"{suspicious_pct:.0f}% of top-3 features are suspicious (lag/calendar)"))

    # Check 5: Economic validity
    valid_feats = sum(1 for f in top5_counts if f in FEATURE_ECONOMICS and f not in SUSPICIOUS_IF_DOMINANT)
    total_feats = len(top5_counts)
    if valid_feats / total_feats > 0.7:
        checks.append(("PASS", f"{valid_feats}/{total_feats} top features have economic justification"))
    else:
        checks.append(("WARN", f"Only {valid_feats}/{total_feats} top features have economic justification"))

    # Print verdict
    for status, msg in checks:
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}[status]
        print(f"  {icon} [{status}] {msg}")

    passes = sum(1 for s, _ in checks if s == "PASS")
    warns = sum(1 for s, _ in checks if s == "WARN")
    fails = sum(1 for s, _ in checks if s == "FAIL")

    print(f"\nScore: {passes} PASS, {warns} WARN, {fails} FAIL out of {len(checks)} checks")

    if fails == 0 and warns <= 1:
        print("\n>>> VERDICT: Patterns appear GENUINE. Proceed to paper trading with confidence.")
    elif fails <= 1 and warns <= 2:
        print("\n>>> VERDICT: Patterns are PARTIALLY validated. Proceed with caution.")
        print("    Monitor feature stability weekly during paper trading.")
    else:
        print("\n>>> VERDICT: Patterns are QUESTIONABLE. Significant overfitting detected.")
        print("    Recommended: collect more data (higher frequency), reduce features,")
        print("    increase regularization before paper trading.")

    print("\n  KEY RECOMMENDATION: The models work but need HIGHER-FREQUENCY DATA")
    print("  (15-min bars) to get enough samples for robust patterns.")
    print("  Daily data (~250 bars) is fundamentally insufficient for reliable ML.")


def main():
    all_meta = load_all_metadata()
    if not all_meta:
        print("No trained models found in data/ml_models/")
        return

    top5, top10 = analyze_cross_ticker_features(all_meta)
    analyze_temporal_stability(all_meta)
    analyze_economic_validity(top5)
    analyze_overfitting_indicators(all_meta)

    # Run individual feature significance on best US and best India ticker
    test_individual_feature_significance("AAPL", "us")
    test_individual_feature_significance("HDFCBANK", "india")

    generate_final_verdict(all_meta, top5)


if __name__ == "__main__":
    main()
