"""
Grounded Response Generation module for @AppleSupport.
Contains:
1. TrivialResponseBaseline: Canned generic template
2. SimpleRetrievalBaseline: BM25/TF-IDF nearest neighbor verbatim historical reply
3. GroundedRAGResponseGenerator: Retrieval-Augmented Generation grounded in historical resolutions
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TrivialResponseBaseline:
    """Trivial baseline: always outputs a fixed canned template."""
    CANONICAL_REPLY = (
        "Thanks for reaching out to Apple Support! We're here to help. "
        "Please DM us your device model, iOS version, and issue details so we can assist you further."
    )

    def generate(self, customer_text: str, intent: str = None, escalation: Dict[str, Any] = None) -> str:
        return self.CANONICAL_REPLY


class SimpleRetrievalBaseline:
    """Simple baseline: verbatim retrieval of nearest neighbor historical resolution."""
    def __init__(self, knowledge_base_path: Optional[str] = None):
        if not knowledge_base_path:
            knowledge_base_path = str(Path(__file__).resolve().parents[2] / "data" / "historical_resolutions.json")
        self.kb_path = knowledge_base_path
        self.records = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
        self.inquiry_vectors = None
        self._load_and_index()

    def _load_and_index(self):
        if not os.path.exists(self.kb_path):
            return
        with open(self.kb_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        inquiries = [r["customer_inquiry"] for r in self.records]
        if inquiries:
            self.inquiry_vectors = self.vectorizer.fit_transform(inquiries)

    def generate(self, customer_text: str, intent: str = None, escalation: Dict[str, Any] = None) -> str:
        if not self.records or self.inquiry_vectors is None:
            return TrivialResponseBaseline.CANONICAL_REPLY
        query_vec = self.vectorizer.transform([customer_text])
        sims = cosine_similarity(query_vec, self.inquiry_vectors)[0]
        best_idx = int(np.argmax(sims))
        return self.records[best_idx]["support_resolution"]


class GroundedRAGResponseGenerator:
    """
    Proposed Production Generator:
    Retrieves top-k historical resolution exemplars and synthesizes an empathetic,
    accurate, brand-aligned response adhering to Apple Support Twitter guidelines.
    """

    INTENT_FALLBACK_TEMPLATES = {
        "hardware_battery": (
            "We want to help with your battery life! Go to Settings > Battery > Battery Health to inspect "
            "maximum capacity and background usage. If your device won't charge, test with an Apple-certified cable "
            "and inspect the charging port. DM us your device model if you need more help."
        ),
        "software_os_update": (
            "We're here to help get your device running smoothly. Check Settings > General > Software Update "
            "for the latest iOS release. If an app is freezing, try restarting your device or reinstalling the app. "
            "DM us your current iOS version so we can investigate further."
        ),
        "account_appleid_icloud": (
            "We can help with your Apple ID and iCloud. You can manage passwords and security options at "
            "iforgot.apple.com or appleid.apple.com. Never share verification codes with anyone. "
            "DM us if you need help navigating account recovery."
        ),
        "connectivity_network": (
            "Let's get your connection back on track. Try toggling Airplane Mode on for 10 seconds, or go to "
            "Settings > General > Reset > Reset Network Settings. For Bluetooth accessories, forget the device and "
            "re-pair. DM us if you're still experiencing connection drops."
        ),
        "billing_subscriptions": (
            "We understand unexpected charges can be frustrating. You can view active subscriptions in "
            "Settings > [Your Name] > Subscriptions, or request refunds directly at reportaproblem.apple.com. "
            "DM us if you need help looking up an unfamiliar charge."
        ),
        "physical_damage_repair": (
            "We're sorry to hear about the damage to your device. You can check repair pricing and schedule an "
            "appointment at an Apple Store or Authorized Service Provider at support.apple.com/repair. "
            "Be sure to back up your device before your visit!"
        ),
        "general_inquiry_features": (
            "We're happy to guide you through your Apple features! Let us know your exact device model and iOS "
            "version so we can provide step-by-step instructions. You can also DM us anytime for assistance."
        )
    }

    def __init__(self, knowledge_base_path: Optional[str] = None):
        if not knowledge_base_path:
            knowledge_base_path = str(Path(__file__).resolve().parents[2] / "data" / "historical_resolutions.json")
        self.kb_path = knowledge_base_path
        self.records = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
        self.inquiry_vectors = None
        self._load_and_index()

    def _load_and_index(self):
        if not os.path.exists(self.kb_path):
            return
        with open(self.kb_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        inquiries = [r["customer_inquiry"] for r in self.records]
        if inquiries:
            self.inquiry_vectors = self.vectorizer.fit_transform(inquiries)

    def retrieve_exemplars(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.records or self.inquiry_vectors is None:
            return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.inquiry_vectors)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "inquiry": self.records[idx]["customer_inquiry"],
                "resolution": self.records[idx]["support_resolution"],
                "score": float(sims[idx])
            })
        return results

    def generate(self, customer_text: str, intent: str, escalation: Dict[str, Any], entities: Dict[str, Any] = None) -> str:
        # Retrieve historical exemplars
        exemplars = self.retrieve_exemplars(customer_text, top_k=3)
        top_match = exemplars[0] if exemplars else None

        should_escalate = escalation.get("escalate", False)
        reason = escalation.get("stated_reason", "")

        # 1. Critical safety hazard handling
        if escalation.get("risk_category") == "acute_safety":
            return (
                "Your safety is our top priority. Please stop using and unplug your device immediately. "
                "We are escalating this to our Safety team. Please DM us your phone number and full name "
                "so a supervisor can reach out right away."
            )

        # 2. Urgent security fraud handling
        if escalation.get("risk_category") == "security_fraud":
            return (
                "We take account security very seriously. Please change your Apple ID password immediately at "
                "appleid.apple.com and ensure two-factor authentication is active. DM us right away so our "
                "Security Specialists can review your account access."
            )

        # 3. Hardware physical damage / store repair escalation
        if should_escalate and intent == "physical_damage_repair":
            return (
                "We're sorry to hear about the damage. Physical repairs require inspection by a certified technician. "
                "You can view pricing and book an appointment at an Apple Store or Authorized Service Provider at "
                "support.apple.com/repair. Please DM us if you'd like help finding local options."
            )

        # 4. General human escalation with stated reason
        if should_escalate:
            escalation_prefix = "We understand how frustrating this is, and we want to ensure this is handled properly. "
            if top_match and top_match["score"] > 0.40:
                core_advice = top_match["resolution"]
                return f"{escalation_prefix}{core_advice} Please DM us your serial number so an advisor can step in."
            return (
                f"{escalation_prefix}Because this requires specialist attention ({reason}), "
                f"please DM us your serial number and contact details so our Senior Support team can assist."
            )

        # 5. High-confidence historical exemplar grounding (Auto-handle)
        if top_match and top_match["score"] >= 0.35:
            resolution = top_match["resolution"]
            # Ensure tone ends with Apple standard DM offer
            if "DM us" not in resolution and "direct message" not in resolution.lower():
                resolution = resolution.rstrip(".") + ". DM us if you need further assistance!"
            return resolution

        # 6. Intent-guided grounded fallback
        fallback = self.INTENT_FALLBACK_TEMPLATES.get(intent, self.INTENT_FALLBACK_TEMPLATES["general_inquiry_features"])
        return fallback
