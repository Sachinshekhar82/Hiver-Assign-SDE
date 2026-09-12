"""
Automated evaluation metrics for intent classification, escalation policy, and response text generation.
"""

from collections import Counter
import math
import re
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

class AutomatedMetricsCalculator:
    @staticmethod
    def evaluate_intent(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
        acc = accuracy_score(y_true, y_pred)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )
        p_per, r_per, f1_per, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

        per_class = {}
        for idx, lbl in enumerate(labels):
            per_class[lbl] = {
                "precision": float(p_per[idx]),
                "recall": float(r_per[idx]),
                "f1": float(f1_per[idx]),
                "support": int(support[idx])
            }

        return {
            "accuracy": float(acc),
            "macro_precision": float(p_macro),
            "macro_recall": float(r_macro),
            "macro_f1": float(f1_macro),
            "per_class": per_class,
            "confusion_matrix": cm,
            "labels": labels
        }

    @staticmethod
    def evaluate_escalation(y_true: List[bool], y_pred: List[bool]) -> Dict[str, Any]:
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and not yp)

        # Critical Safety Metric: False Auto-Handle Rate (FAHR) = FN / (FN + TP)
        # Fraction of dangerous/escalation cases mistakenly auto-handled
        fahr = (fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        # Cost Metric: False Escalation Rate (FER) = FP / (FP + TN)
        # Fraction of routine cases unnecessarily escalated to expensive human agents
        fer = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        return {
            "accuracy": float(acc),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "false_auto_handle_rate_FAHR": float(fahr),
            "false_escalation_rate_FER": float(fer)
        }

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    @classmethod
    def compute_rouge(cls, hyp: str, ref: str) -> Dict[str, float]:
        hyp_tokens = cls._tokenize(hyp)
        ref_tokens = cls._tokenize(ref)

        if not hyp_tokens or not ref_tokens:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        # ROUGE-1
        hyp_c1 = Counter(hyp_tokens)
        ref_c1 = Counter(ref_tokens)
        overlap1 = sum(min(hyp_c1[w], ref_c1[w]) for w in hyp_c1)
        r1_r = overlap1 / len(ref_tokens)
        r1_p = overlap1 / len(hyp_tokens)
        r1_f = (2 * r1_p * r1_r / (r1_p + r1_r)) if (r1_p + r1_r) > 0 else 0.0

        # ROUGE-2
        hyp_bigrams = [f"{hyp_tokens[i]} {hyp_tokens[i+1]}" for i in range(len(hyp_tokens)-1)]
        ref_bigrams = [f"{ref_tokens[i]} {ref_tokens[i+1]}" for i in range(len(ref_tokens)-1)]
        if not ref_bigrams:
            r2_f = 0.0
        else:
            hyp_c2 = Counter(hyp_bigrams)
            ref_c2 = Counter(ref_bigrams)
            overlap2 = sum(min(hyp_c2[w], ref_c2[w]) for w in hyp_c2)
            r2_r = overlap2 / len(ref_bigrams)
            r2_p = (overlap2 / len(hyp_bigrams)) if hyp_bigrams else 0.0
            r2_f = (2 * r2_p * r2_r / (r2_p + r2_r)) if (r2_p + r2_r) > 0 else 0.0

        # ROUGE-L (Longest Common Subsequence)
        m, n = len(ref_tokens), len(hyp_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[m][n]
        rl_r = lcs / m
        rl_p = lcs / n
        rl_f = (2 * rl_p * rl_r / (rl_p + rl_r)) if (rl_p + rl_r) > 0 else 0.0

        return {
            "rouge1": float(r1_f),
            "rouge2": float(r2_f),
            "rougeL": float(rl_f)
        }

    @classmethod
    def compute_bleu4(cls, hyp: str, ref: str) -> float:
        hyp_tokens = cls._tokenize(hyp)
        ref_tokens = cls._tokenize(ref)
        if len(hyp_tokens) < 4 or len(ref_tokens) < 4:
            return 0.0

        # Brevity penalty
        c = len(hyp_tokens)
        r = len(ref_tokens)
        bp = 1.0 if c > r else math.exp(1 - r / c)

        precisions = []
        for n in range(1, 5):
            hyp_ngrams = [tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1)]
            ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)]
            hyp_c = Counter(hyp_ngrams)
            ref_c = Counter(ref_ngrams)
            overlap = sum(min(hyp_c[ng], ref_c[ng]) for ng in hyp_c)
            p_n = overlap / len(hyp_ngrams) if hyp_ngrams else 0.0
            if p_n == 0:
                return 0.0
            precisions.append(p_n)

        score = bp * math.exp(sum(math.log(p) for p in precisions) / 4)
        return float(score)

    @classmethod
    def evaluate_text_generation(cls, hypotheses: List[str], references: List[str]) -> Dict[str, float]:
        r1_list, r2_list, rl_list, bleu_list = [], [], [], []
        for h, r in zip(hypotheses, references):
            rouge = cls.compute_rouge(h, r)
            r1_list.append(rouge["rouge1"])
            r2_list.append(rouge["rouge2"])
            rl_list.append(rouge["rougeL"])
            bleu_list.append(cls.compute_bleu4(h, r))

        return {
            "mean_rouge1": float(np.mean(r1_list)),
            "mean_rouge2": float(np.mean(r2_list)),
            "mean_rougeL": float(np.mean(rl_list)),
            "mean_bleu4": float(np.mean(bleu_list))
        }
