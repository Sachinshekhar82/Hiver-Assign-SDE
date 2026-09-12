"""
Intent Classification module for @AppleSupport agent.
Contains:
1. MajorityClassBaseline: Trivial baseline
2. TfidfBaseline: Simple machine learning baseline (TF-IDF + LogisticRegression)
3. CalibratedHybridClassifier: Proposed multi-channel semantic classifier
"""

import re
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

INTENT_LABELS = [
    "hardware_battery",
    "software_os_update",
    "account_appleid_icloud",
    "connectivity_network",
    "billing_subscriptions",
    "physical_damage_repair",
    "general_inquiry_features"
]

class MajorityClassBaseline:
    """Trivial baseline that always predicts the majority class."""
    def __init__(self, default_class: str = "general_inquiry_features"):
        self.default_class = default_class

    def fit(self, texts: List[str], labels: List[str]):
        if labels:
            from collections import Counter
            counts = Counter(labels)
            self.default_class = counts.most_common(1)[0][0]

    def predict(self, text: str) -> str:
        return self.default_class

    def predict_batch(self, texts: List[str]) -> List[str]:
        return [self.default_class for _ in texts]


class TfidfBaseline:
    """Simple ML baseline using TF-IDF n-grams + Logistic Regression."""
    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=2500, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42))
        ])
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]):
        self.pipeline.fit(texts, labels)
        self.is_fitted = True

    def predict(self, text: str) -> str:
        if not self.is_fitted:
            return "general_inquiry_features"
        return self.pipeline.predict([text])[0]

    def predict_batch(self, texts: List[str]) -> List[str]:
        if not self.is_fitted:
            return ["general_inquiry_features"] * len(texts)
        return list(self.pipeline.predict(texts))

    def predict_proba(self, text: str) -> Dict[str, float]:
        if not self.is_fitted:
            return {cls: 1.0 / len(INTENT_LABELS) for cls in INTENT_LABELS}
        probs = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        return {cls: float(prob) for cls, prob in zip(classes, probs)}


class CalibratedHybridClassifier:
    """
    Proposed Production Classifier:
    Combines calibrated TF-IDF representation, high-salience technical regex rules,
    and domain feature boosts with softmax temperature calibration.
    """

    INTENT_RULES = {
        "hardware_battery": [
            r"\bbatter(?:y|ies)\b", r"\bdrain(?:ing)?\b", r"\bcharg(?:e|ing|er)\b",
            r"\bbulg(?:e|ing)\b", r"\bswollen\b", r"\blo[w\s]+power\s+mode\b",
            r"\bmagsafe\b", r"\boverheat(?:ing)?\b", r"\bhot\b", r"\bshut\s*off\b"
        ],
        "software_os_update": [
            r"\bios\b", r"\bupdate\b", r"\bupdating\b", r"\binstall(?:ing)?\b",
            r"\bcrash(?:es|ing)?\b", r"\bboot\s*loop\b", r"\bapple\s*logo\b",
            r"\bfreez(?:e|ing|ed)\b", r"\bverifying\s*update\b", r"\b4013\b",
            r"\bsystem\s*storage\b", r"\bother\s*storage\b", r"\bkeyboard\s*lag\b"
        ],
        "account_appleid_icloud": [
            r"\bapple\s*id\b", r"\bicloud\b", r"\bpasscode\b", r"\bpassword\b",
            r"\btwo[- ]factor\b", r"\b2fa\b", r"\bverification\s*code\b",
            r"\blocked\b", r"\bhack(?:ed)?\b", r"\bunlock\b", r"\biforgot\b",
            r"\bactivation\s*lock\b", r"\bfind\s*my\b", r"\bkeychain\b"
        ],
        "connectivity_network": [
            r"\bwi[- ]?fi\b", r"\bbluetooth\b", r"\bno\s*sim\b", r"\bcellular\b",
            r"\bairpods?\b", r"\bhotspot\b", r"\bsearching\.\.\.\b", r"\blte\b",
            r"\b3g\b", r"\b4g\b", r"\bairdrop\b", r"\bgps\b", r"\broaming\b",
            r"\bcarrier\b", r"\bsim\s*card\b"
        ],
        "billing_subscriptions": [
            r"\bcharg(?:ed|e)\b.*\$(?:[0-9]+)", r"\$(?:[0-9]+)", r"\brefund\b",
            r"\bsubscription\b", r"\bapple\s*music\b", r"\bapp\s*store\s*purchase\b",
            r"\bitunes\.com/bill\b", r"\bcredit\s*card\b", r"\bapple\s*pay\b",
            r"\bbilling\b", r"\bin[- ]app\s*purchase\b", r"\bdeclined\b"
        ],
        "physical_damage_repair": [
            r"\bcrack(?:ed)?\b", r"\bshatter(?:ed)?\b", r"\bscreen\s*repair\b",
            r"\bglass\b", r"\bwater\s*damage\b", r"\bdropped\b", r"\bpool\b",
            r"\bgenius\s*bar\b", r"\bapplecare\b", r"\brepair\b", r"\bbroken\b",
            r"\bmicrophone\b", r"\bspeaker\b", r"\btaptic\b", r"\bbutton\s*fell\b"
        ],
        "general_inquiry_features": [
            r"\bpadlock\b", r"\borientation\s*lock\b", r"\bdelete\s*apps?\b",
            r"\bhow\s*do\s*i\b", r"\btrade[- ]in\b", r"\btrue\s*tone\b",
            r"\bface\s*id\b", r"\bscreen\s*recording\b", r"\bdo\s*not\s*disturb\b",
            r"\bmove\s*to\s*ios\b", r"\bscreenshot\b", r"\bnight\s*shift\b",
            r"\bsafari\b", r"\bnotes\b", r"\boffload\b"
        ]
    }

    def __init__(self, temperature: float = 0.85):
        self.temperature = temperature
        self.ml_classifier = TfidfBaseline()
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]):
        self.ml_classifier.fit(texts, labels)
        self.is_fitted = True

    def _rule_scores(self, text: str) -> Dict[str, float]:
        lower = text.lower()
        scores = {intent: 0.0 for intent in INTENT_LABELS}
        for intent, patterns in self.INTENT_RULES.items():
            for pat in patterns:
                matches = len(re.findall(pat, lower))
                if matches > 0:
                    scores[intent] += matches * 1.5
        return scores

    def predict_proba(self, text: str) -> Dict[str, float]:
        rule_scores = self._rule_scores(text)
        ml_probs = self.ml_classifier.predict_proba(text) if self.is_fitted else {cls: 1.0 / len(INTENT_LABELS) for cls in INTENT_LABELS}

        # Combine ML probability distribution with domain rules
        combined_logits = {}
        for intent in INTENT_LABELS:
            # log probability + rule boost
            prob = max(ml_probs.get(intent, 1e-6), 1e-6)
            log_prob = np.log(prob)
            logit = (log_prob + rule_scores.get(intent, 0.0) * 1.2) / self.temperature
            combined_logits[intent] = logit

        # Calibrated softmax
        max_logit = max(combined_logits.values())
        exp_logits = {k: np.exp(v - max_logit) for k, v in combined_logits.items()}
        sum_exp = sum(exp_logits.values())
        calibrated_probs = {k: float(v / sum_exp) for k, v in exp_logits.items()}
        return calibrated_probs

    def predict(self, text: str) -> Tuple[str, float]:
        probs = self.predict_proba(text)
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_intent, top_conf = sorted_probs[0]
        return top_intent, top_conf

    def predict_batch(self, texts: List[str]) -> List[str]:
        return [self.predict(t)[0] for t in texts]
