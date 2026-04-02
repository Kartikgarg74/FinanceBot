"""SHAP-based pattern discovery and validation.

Answers the key question: "What patterns did the model learn, and are they legit?"
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """Analyze model patterns using SHAP values and statistical tests."""

    def __init__(self, output_dir: str | Path = "data/shap_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, model, X: pd.DataFrame, model_name: str = "model") -> dict:
        """
        Run full SHAP analysis on a trained model.

        Returns dict with:
        - global_importance: {feature: mean_abs_shap}
        - top_features: sorted list of (feature, importance)
        - interaction_summary: key feature interactions
        """
        import shap

        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, X)

        shap_values = explainer(X)

        # For multiclass, shap_values.values has shape (n_samples, n_features, n_classes)
        # Take mean absolute across classes for global importance
        if len(shap_values.values.shape) == 3:
            abs_shap = np.abs(shap_values.values).mean(axis=2)  # Average across classes
        else:
            abs_shap = np.abs(shap_values.values)

        # Global feature importance
        mean_abs_shap = abs_shap.mean(axis=0)
        feature_names = X.columns.tolist()
        importance = dict(zip(feature_names, mean_abs_shap))
        importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

        top_features = list(importance.items())[:15]

        result = {
            "global_importance": importance,
            "top_features": top_features,
            "shap_values": shap_values,
            "model_name": model_name,
        }

        logger.info("[%s] Top 5 features by SHAP: %s",
                     model_name,
                     ", ".join(f"{name}={val:.4f}" for name, val in top_features[:5]))

        return result

    def generate_report(self, shap_result: dict, X: pd.DataFrame) -> str:
        """Generate a human-readable text report of discovered patterns."""
        lines = []
        model_name = shap_result["model_name"]
        top = shap_result["top_features"]

        lines.append(f"SHAP Pattern Report — {model_name}")
        lines.append("=" * 50)
        lines.append("")
        lines.append("Top Features by Mean |SHAP| Value:")
        lines.append("-" * 40)

        for rank, (feat, imp) in enumerate(top, 1):
            lines.append(f"  {rank:2d}. {feat:<30s} {imp:.4f}")

        lines.append("")
        lines.append("Feature Statistics:")
        lines.append("-" * 40)

        for feat, imp in top[:10]:
            if feat in X.columns:
                col = X[feat].dropna()
                lines.append(f"  {feat}: mean={col.mean():.4f}, std={col.std():.4f}, "
                             f"min={col.min():.4f}, max={col.max():.4f}")

        # Save report
        report_path = self.output_dir / f"shap_report_{model_name}.txt"
        report_text = "\n".join(lines)
        report_path.write_text(report_text)
        logger.info("SHAP report saved: %s", report_path)

        return report_text

    def save_plots(self, shap_result: dict, X: pd.DataFrame):
        """Save SHAP summary and dependence plots."""
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        model_name = shap_result["model_name"]
        shap_values = shap_result["shap_values"]

        # Summary plot
        plt.figure(figsize=(12, 8))
        # For multiclass, use class index 2 (BUY class) for the summary plot
        if len(shap_values.values.shape) == 3:
            shap.summary_plot(shap_values.values[:, :, 2], X,
                              show=False, max_display=20)
        else:
            shap.summary_plot(shap_values, X, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"shap_summary_{model_name}.png", dpi=150)
        plt.close()

        # Dependence plots for top 5 features
        top_features = [f for f, _ in shap_result["top_features"][:5]]
        for feat in top_features:
            if feat in X.columns:
                plt.figure(figsize=(8, 5))
                feat_idx = X.columns.tolist().index(feat)
                if len(shap_values.values.shape) == 3:
                    vals = shap_values.values[:, feat_idx, 2]  # BUY class
                else:
                    vals = shap_values.values[:, feat_idx]
                plt.scatter(X[feat].values, vals, alpha=0.5, s=10)
                plt.xlabel(feat)
                plt.ylabel(f"SHAP value ({feat})")
                plt.title(f"SHAP Dependence: {feat}")
                plt.tight_layout()
                plt.savefig(self.output_dir / f"shap_dep_{model_name}_{feat}.png", dpi=150)
                plt.close()

        logger.info("SHAP plots saved to %s", self.output_dir)


class PatternValidator:
    """Statistical validation of discovered trading patterns."""

    def validate_returns(self, strategy_returns: np.ndarray) -> dict:
        """
        Test if strategy returns are statistically significant.

        Returns dict with test results and verdicts.
        """
        results = {}

        # 1. T-test: are mean returns significantly > 0?
        t_stat, p_value = stats.ttest_1samp(strategy_returns, 0)
        results["t_test"] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }

        # 2. Sharpe ratio
        if strategy_returns.std() > 0:
            daily_sharpe = strategy_returns.mean() / strategy_returns.std()
            annualized_sharpe = daily_sharpe * np.sqrt(252)
        else:
            annualized_sharpe = 0
        results["sharpe_ratio"] = float(annualized_sharpe)

        # 3. Monte Carlo permutation test
        n_permutations = 5000
        actual_sharpe = annualized_sharpe
        random_sharpes = []
        for _ in range(n_permutations):
            shuffled = np.random.permutation(strategy_returns)
            if shuffled.std() > 0:
                rs = shuffled.mean() / shuffled.std() * np.sqrt(252)
            else:
                rs = 0
            random_sharpes.append(rs)

        mc_p_value = np.mean(np.array(random_sharpes) >= actual_sharpe)
        results["monte_carlo"] = {
            "actual_sharpe": float(actual_sharpe),
            "p_value": float(mc_p_value),
            "significant": mc_p_value < 0.05,
            "percentile": float(100 * (1 - mc_p_value)),
        }

        # 4. Max drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_dd = float(drawdown.min())
        results["max_drawdown"] = max_dd

        # 5. Profit factor
        wins = strategy_returns[strategy_returns > 0].sum()
        losses = abs(strategy_returns[strategy_returns < 0].sum())
        results["profit_factor"] = float(wins / losses) if losses > 0 else float("inf")

        # 6. Win rate
        n_trades = len(strategy_returns[strategy_returns != 0])
        n_wins = len(strategy_returns[strategy_returns > 0])
        results["win_rate"] = float(n_wins / n_trades) if n_trades > 0 else 0

        # Overall verdict
        results["verdict"] = self._verdict(results)

        return results

    def validate_feature_stability(
        self, fold_importances: list[dict[str, float]], top_n: int = 10
    ) -> dict:
        """
        Check if the same features are important across all walk-forward folds.

        fold_importances: list of {feature: importance} dicts, one per fold.
        """
        if not fold_importances:
            return {"stable": False, "reason": "No folds to compare"}

        # Get top-N features from each fold
        top_per_fold = []
        for imp in fold_importances:
            sorted_feats = sorted(imp.items(), key=lambda x: -x[1])
            top_per_fold.append(set(f for f, _ in sorted_feats[:top_n]))

        # Jaccard similarity between all pairs
        similarities = []
        for i in range(len(top_per_fold)):
            for j in range(i + 1, len(top_per_fold)):
                intersection = len(top_per_fold[i] & top_per_fold[j])
                union = len(top_per_fold[i] | top_per_fold[j])
                similarities.append(intersection / union if union > 0 else 0)

        mean_sim = np.mean(similarities) if similarities else 0

        # Features that appear in ALL folds' top-N
        common_features = set.intersection(*top_per_fold) if top_per_fold else set()

        return {
            "stable": mean_sim > 0.5,
            "mean_jaccard_similarity": float(mean_sim),
            "common_top_features": sorted(common_features),
            "n_common": len(common_features),
        }

    def _verdict(self, results: dict) -> str:
        """Generate human-readable verdict."""
        checks = []

        if results["t_test"]["significant"]:
            checks.append("PASS: Returns statistically significant (p<0.05)")
        else:
            checks.append(f"FAIL: Returns not significant (p={results['t_test']['p_value']:.3f})")

        if results["monte_carlo"]["significant"]:
            checks.append(f"PASS: Monte Carlo test (top {results['monte_carlo']['percentile']:.1f}%)")
        else:
            checks.append("FAIL: Monte Carlo test — ordering doesn't matter")

        if results["sharpe_ratio"] > 1.0:
            checks.append(f"PASS: Sharpe ratio {results['sharpe_ratio']:.2f} > 1.0")
        else:
            checks.append(f"WARN: Sharpe ratio {results['sharpe_ratio']:.2f} < 1.0")

        if results["max_drawdown"] > -0.20:
            checks.append(f"PASS: Max drawdown {results['max_drawdown']:.1%} within 20% limit")
        else:
            checks.append(f"FAIL: Max drawdown {results['max_drawdown']:.1%} exceeds 20%")

        if results["profit_factor"] > 1.3:
            checks.append(f"PASS: Profit factor {results['profit_factor']:.2f} > 1.3")
        else:
            checks.append(f"WARN: Profit factor {results['profit_factor']:.2f} < 1.3")

        return "\n".join(checks)
