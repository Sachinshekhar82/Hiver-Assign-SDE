"""
Human-Judge Agreement Validation Module.
Computes inter-rater reliability statistics between Human Annotator ratings and LLM-as-a-Judge scores:
- Cohen's Quadratic Weighted Kappa (kappa_w)
- Pearson Correlation Coefficient (r)
- Spearman Rank Correlation Coefficient (rho)
- Mean Absolute Error (MAE)
- Exact Agreement Rate (%)
- Adjacent Agreement Rate (within 1 point) (%)
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

class HumanJudgeAgreementAnalyzer:
    @staticmethod
    def compute_agreement(human_scores: List[float], judge_scores: List[float]) -> Dict[str, Any]:
        assert len(human_scores) == len(judge_scores), "Lengths of scores must match."
        n = len(human_scores)
        if n == 0:
            return {}

        # Round continuous composite scores to integer 1-5 for discrete agreement metrics
        human_discrete = [int(round(s)) for s in human_scores]
        judge_discrete = [int(round(s)) for s in judge_scores]

        # 1. Quadratic Weighted Kappa
        # Measures agreement factoring in magnitude of disagreement
        qw_kappa = cohen_kappa_score(human_discrete, judge_discrete, weights="quadratic")

        # 2. Linear Correlation (Pearson)
        pearson_corr, pearson_p = pearsonr(human_scores, judge_scores)

        # 3. Rank Correlation (Spearman)
        spearman_corr, spearman_p = spearmanr(human_scores, judge_scores)

        # 4. Mean Absolute Error (MAE)
        mae = float(np.mean(np.abs(np.array(human_scores) - np.array(judge_scores))))

        # 5. Exact and Within-1 Agreement
        exact_matches = sum(1 for h, j in zip(human_discrete, judge_discrete) if h == j)
        within_one_matches = sum(1 for h, j in zip(human_discrete, judge_discrete) if abs(h - j) <= 1)

        exact_rate = exact_matches / n
        adjacent_rate = within_one_matches / n

        return {
            "sample_size": n,
            "quadratic_weighted_kappa": float(qw_kappa),
            "pearson_correlation": float(pearson_corr),
            "pearson_p_value": float(pearson_p),
            "spearman_correlation": float(spearman_corr),
            "spearman_p_value": float(spearman_p),
            "mean_absolute_error_MAE": float(mae),
            "exact_agreement_rate": float(exact_rate),
            "within_1_point_agreement_rate": float(adjacent_rate),
            "human_mean_score": float(np.mean(human_scores)),
            "judge_mean_score": float(np.mean(judge_scores))
        }

    @classmethod
    def format_summary_table(cls, stats: Dict[str, Any]) -> str:
        lines = [
            "=================================================================",
            "             HUMAN vs. LLM-AS-A-JUDGE AGREEMENT REPORT           ",
            "=================================================================",
            f"Evaluated Test Pairs                : {stats.get('sample_size', 0)}",
            f"Cohen's Quadratic Weighted Kappa (kw): {stats.get('quadratic_weighted_kappa', 0.0):.4f}",
            f"Pearson Linear Correlation (r)      : {stats.get('pearson_correlation', 0.0):.4f} (p = {stats.get('pearson_p_value', 0.0):.2e})",
            f"Spearman Rank Correlation (rho)     : {stats.get('spearman_correlation', 0.0):.4f} (p = {stats.get('spearman_p_value', 0.0):.2e})",
            f"Mean Absolute Error (MAE)           : {stats.get('mean_absolute_error_MAE', 0.0):.4f} stars",
            f"Exact Rating Agreement Rate         : {stats.get('exact_agreement_rate', 0.0)*100:.1f}%",
            f"Within-1-Point Agreement Rate       : {stats.get('within_1_point_agreement_rate', 0.0)*100:.1f}%",
            f"Human Mean Score / Judge Mean Score : {stats.get('human_mean_score', 0.0):.2f} / {stats.get('judge_mean_score', 0.0):.2f}",
            "================================================================="
        ]
        return "\n".join(lines)
