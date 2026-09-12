"""
Escalation Decision Engine for @AppleSupport.
Decides whether an incoming customer message should be auto-handled or escalated to a human agent,
along with an explicit stated reason and calibrated confidence score.
"""

import re
from typing import Dict, Any, Tuple

class TrivialEscalationBaseline:
    """Trivial baseline: never escalates, always auto-handles."""
    def evaluate(self, text: str, intent: str = None) -> Dict[str, Any]:
        return {
            "escalate": False,
            "decision_label": "AUTO_HANDLE",
            "stated_reason": "Trivial baseline default: always auto-handle without inspection.",
            "confidence": 0.5,
            "risk_category": "none"
        }

class SimpleRegexEscalationBaseline:
    """Simple baseline: regex keyword matching for obvious escalation terms."""
    KEYWORDS = [
        r"\bbroken\b", r"\brepair\b", r"\bshatter(?:ed)?\b", r"\bhack(?:ed)?\b",
        r"\bstolen\b", r"\brefund\b", r"\blawsuit\b", r"\blegal\b", r"\bwater\b",
        r"\bsmoke\b", r"\bspark\b", r"\bexplode\b"
    ]

    def evaluate(self, text: str, intent: str = None) -> Dict[str, Any]:
        lower = text.lower()
        matched = []
        for kw in self.KEYWORDS:
            if re.search(kw, lower):
                matched.append(kw.replace(r"\b", "").replace("(?:ed)?", ""))

        if matched:
            return {
                "escalate": True,
                "decision_label": "ESCALATE",
                "stated_reason": f"Simple keyword match triggered: {', '.join(matched)}.",
                "confidence": 0.70,
                "risk_category": "keyword_match"
            }
        return {
            "escalate": False,
            "decision_label": "AUTO_HANDLE",
            "stated_reason": "No high-risk keywords detected; routing to standard auto-handling.",
            "confidence": 0.65,
            "risk_category": "none"
        }

class MultiFactorEscalationGuardrail:
    """
    Proposed Production Escalation Engine:
    Multi-tier policy engine evaluating physical safety, hardware viability,
    account compromise, financial disputes, and customer frustration churn.
    """

    CRITICAL_SAFETY_PATTERNS = [
        (r"\b(?:bulg\w*|swollen|expand\w*)\b.*\bbatter\w*\b|\bbatter\w*\b.*\b(?:bulg\w*|swollen|expand\w*)\b",
         "Acute physical safety hazard: swollen/bulging battery poses thermal runaway risk."),
        (r"\b(?:spark\w*|fire|smoke|burning\s*smell|burnt\s*plastic)\b",
         "Electrical safety hazard: sparking adapter or burning smell requires immediate safety protocol."),
        (r"\b(?:battery\s*swollen\s*so\s*bad\s*it\s*cracked)\b",
         "Severe structural battery expansion compromising chassis integrity.")
    ]

    HARDWARE_DAMAGE_PATTERNS = [
        (r"\b(?:shatter\w*|cracked?\s*glass|front\s*glass|loose\s*glass|glass\s*shards)\b",
         "Physical enclosure damage: shattered glass requires technician repair or whole-unit replacement."),
        (r"\b(?:fell\s*into\s*(?:the\s*)?(?:pool|toilet|water|sink)|liquid|corrosion|soaked)\b",
         "Liquid ingress with component failure: requires physical teardown and liquid contact inspection."),
        (r"\b(?:greyed\s*out|grayed\s*out)\s*wi[- ]?fi\b",
         "Wi-Fi baseband IC / antenna hardware failure: requires logic board repair at Apple Store."),
        (r"\b(?:green\s*line|vertical\s*line)\b.*(?:oled|screen|display)",
         "Display hardware defect: damaged OLED subpixel driver lines necessitate screen assembly replacement."),
        (r"\b(?:volume\s*button\s*fell\s*out|sim\s*tray\s*(?:stuck|broke)|pin\s*broken\s*inside)\b",
         "Chassis mechanical defect: broken physical components require technician extraction."),
        (r"\b(?:unresponsive\s*to\s*touch|touch\s*remains\s*completely\s*dead|black\s*camera\s*preview)\b",
         "Hardware digitizer or camera sensor failure: unresolvable via software restart."),
        (r"\b(?:kernel\s*panic|error\s*4013|infinite\s*bootloop|watchdog\s*timer)\b",
         "Unrecoverable logic board firmware/NAND failure: requires authorized diagnostic restore or swap.")
    ]

    SECURITY_FRAUD_PATTERNS = [
        (r"\b(?:hacked|unauthorized\s*(?:login|device|access)|someone\b.*(?:logged\s*into|changed|access)|(?:logged\s*into|changed)\s*my\s*(?:apple\s*id|password|phone|account))\b",
         "Account security breach: unauthorized access requires tier-2 identity verification and lock."),
        (r"\b(?:stolen\s*credit\s*card|someone\s*is\s*buying\s*thousands)\b",
         "Active financial fraud: card stolen with live fraudulent transactions."),
        (r"\b(?:deceased|died|death\s*certificate|legacy\s*contact)\b",
         "Estate / deceased account transfer: requires formal legal documentation and human verification."),
        (r"\b(?:lost\s*my\s*trusted\s*phone\s*number\s*and\s*don'?t\s*have\s*any\s*other)\b",
         "Total loss of authentication factors: manual Account Recovery escalation required.")
    ]

    BILLING_DISPUTE_PATTERNS = [
        (r"\b(?:charged\s*me\s*three\s*times|charged\s*twice|double\s*bill(?:ing)?)\b",
         "Billing ledger discrepancy: duplicate transaction requires human agent review and refund credit."),
        (r"\b(?:refund\s*request\s*[0-9]+\s*days\s*ago|promised\s*a\s*refund)\b",
         "SLA violation: unresolved refund beyond standard processing window requires human investigation."),
        (r"\b(?:threaten(?:ing)?\s*legal\s*action|lawyer|attorney|sue\b)\b",
         "Legal threat / formal complaint: mandatory escalation to senior relations supervisor.")
    ]

    FRUSTRATION_EXHAUSTION_PATTERNS = [
        (r"\b(?:already\s*tried\s*(?:resetting|restarting|3\s*cables|everything)|done\s*that\s*[0-9]+\s*times)\b",
         "Customer troubleshooting exhaustion: client has already exhausted self-service workflows without success."),
        (r"\b(?:unacceptable|refused\s*to\s*help|bricked\s*my\s*device)\b",
         "Critical customer dissatisfaction: high churn risk requires empathetic senior advisor touch.")
    ]

    def evaluate(self, text: str, intent: str = None) -> Dict[str, Any]:
        lower = text.lower()

        # 1. Critical Safety Check
        for pat, reason in self.CRITICAL_SAFETY_PATTERNS:
            if re.search(pat, lower):
                return {
                    "escalate": True,
                    "decision_label": "ESCALATE",
                    "stated_reason": reason,
                    "confidence": 0.98,
                    "risk_category": "acute_safety"
                }

        # 2. Security & Account Hijack Check
        for pat, reason in self.SECURITY_FRAUD_PATTERNS:
            if re.search(pat, lower):
                return {
                    "escalate": True,
                    "decision_label": "ESCALATE",
                    "stated_reason": reason,
                    "confidence": 0.95,
                    "risk_category": "security_fraud"
                }

        # 3. Hardware Physical Damage Check
        for pat, reason in self.HARDWARE_DAMAGE_PATTERNS:
            if re.search(pat, lower):
                return {
                    "escalate": True,
                    "decision_label": "ESCALATE",
                    "stated_reason": reason,
                    "confidence": 0.93,
                    "risk_category": "hardware_failure"
                }

        # 4. Financial & Billing Dispute Check
        for pat, reason in self.BILLING_DISPUTE_PATTERNS:
            if re.search(pat, lower):
                return {
                    "escalate": True,
                    "decision_label": "ESCALATE",
                    "stated_reason": reason,
                    "confidence": 0.92,
                    "risk_category": "billing_dispute"
                }

        # 5. Customer Frustration & Exhaustion Check
        for pat, reason in self.FRUSTRATION_EXHAUSTION_PATTERNS:
            if re.search(pat, lower):
                return {
                    "escalate": True,
                    "decision_label": "ESCALATE",
                    "stated_reason": reason,
                    "confidence": 0.88,
                    "risk_category": "customer_frustration"
                }

        # 6. Intent-specific policy defaults
        if intent == "physical_damage_repair":
            if not re.search(r"\b(?:how\s*much|cost|pricing|what\s*to\s*bring|do\s*i\s*need\s*to\s*back\s*up|loaner)\b", lower):
                return {
                    "escalate": True,
                    "decision_label": "ESCALATE",
                    "stated_reason": "Physical repair requirement: service inspection needed at Apple Authorized Service Provider.",
                    "confidence": 0.85,
                    "risk_category": "hardware_repair"
                }

        # Safe for automated self-service handling
        return {
            "escalate": False,
            "decision_label": "AUTO_HANDLE",
            "stated_reason": "Issue is safely resolvable through standard self-service troubleshooting or guided settings configuration.",
            "confidence": 0.91,
            "risk_category": "safe_self_service"
        }
