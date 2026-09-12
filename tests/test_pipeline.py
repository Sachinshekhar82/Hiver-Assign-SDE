"""
Unit and integration tests for @AppleSupport AI Agent.
"""

import pytest
from src.data.preprocessor import TweetPreprocessor
from src.agent.intent_classifier import (
    MajorityClassBaseline,
    TfidfBaseline,
    CalibratedHybridClassifier,
    INTENT_LABELS
)
from src.agent.escalation_engine import (
    TrivialEscalationBaseline,
    SimpleRegexEscalationBaseline,
    MultiFactorEscalationGuardrail
)
from src.agent.response_generator import GroundedRAGResponseGenerator
from src.agent.orchestrator import SupportAgentOrchestrator
from src.eval.metrics import AutomatedMetricsCalculator
from src.eval.human_agreement import HumanJudgeAgreementAnalyzer

def test_preprocessor():
    raw = "@AppleSupport my iPhone 7 has battery draining fast on iOS 11.0.2! http://t.co/xyz"
    clean = TweetPreprocessor.clean_text(raw)
    assert "@AppleSupport" not in clean
    assert "http" not in clean
    assert "iPhone 7" in clean

    entities = TweetPreprocessor.extract_entities(raw)
    assert entities["has_device_info"] is True
    assert entities["has_os_info"] is True

def test_intent_classifiers():
    train_texts = [
        "battery draining fast", "ios 11 update stuck", "apple id password locked",
        "wifi greyed out and no sim", "charged twice for apple music",
        "cracked screen glass broken", "how do i turn on portrait orientation lock"
    ]
    train_labels = [
        "hardware_battery", "software_os_update", "account_appleid_icloud",
        "connectivity_network", "billing_subscriptions",
        "physical_damage_repair", "general_inquiry_features"
    ]

    # Baseline 1
    t_clf = MajorityClassBaseline()
    t_clf.fit(train_texts, train_labels)
    assert t_clf.predict("any text") in INTENT_LABELS

    # Baseline 2
    s_clf = TfidfBaseline()
    s_clf.fit(train_texts, train_labels)
    pred = s_clf.predict("battery life low")
    assert pred in INTENT_LABELS

    # Proposed
    p_clf = CalibratedHybridClassifier()
    p_clf.fit(train_texts, train_labels)
    pred, conf = p_clf.predict("my battery is draining in two hours")
    assert pred == "hardware_battery"
    assert 0.0 <= conf <= 1.0

def test_escalation_engine_safety():
    engine = MultiFactorEscalationGuardrail()

    # Acute safety hazard
    res_safety = engine.evaluate("my battery is bulging and pushing the screen off")
    assert res_safety["escalate"] is True
    assert res_safety["risk_category"] == "acute_safety"
    assert "safety" in res_safety["stated_reason"].lower()

    # Hardware damage
    res_hw = engine.evaluate("shattered screen with glass falling out")
    assert res_hw["escalate"] is True
    assert res_hw["risk_category"] == "hardware_failure"

    # Security fraud
    res_sec = engine.evaluate("someone in russia logged into my apple id and changed my password")
    assert res_sec["escalate"] is True
    assert res_sec["risk_category"] == "security_fraud"

    # Routine software troubleshooting
    res_routine = engine.evaluate("how do I get rid of the padlock icon on screen")
    assert res_routine["escalate"] is False
    assert res_routine["risk_category"] == "safe_self_service"

def test_response_generator():
    generator = GroundedRAGResponseGenerator()
    # Test acute safety response
    reply_safety = generator.generate(
        "battery is bulging",
        intent="hardware_battery",
        escalation={"escalate": True, "risk_category": "acute_safety", "stated_reason": "thermal hazard"}
    )
    assert "safety" in reply_safety.lower()
    assert "unplug" in reply_safety.lower() or "stop using" in reply_safety.lower()

def test_orchestrator_integration():
    orchestrator = SupportAgentOrchestrator(mode="proposed")
    res = orchestrator.process_message("My iPhone 7 battery health is 75%, how do I replace it?")
    assert "intent" in res
    assert "escalation" in res
    assert "reply" in res
    assert res["intent"] == "hardware_battery"
    assert len(res["reply"]) > 20

def test_metrics_calculation():
    y_true = ["hardware_battery", "software_os_update"]
    y_pred = ["hardware_battery", "general_inquiry_features"]
    res = AutomatedMetricsCalculator.evaluate_intent(y_true, y_pred, labels=INTENT_LABELS)
    assert res["accuracy"] == 0.5

    esc_true = [True, False, True]
    esc_pred = [True, False, False]
    esc_res = AutomatedMetricsCalculator.evaluate_escalation(esc_true, esc_pred)
    assert esc_res["true_positives"] == 1
    assert esc_res["false_negatives"] == 1
    assert esc_res["false_auto_handle_rate_FAHR"] == 0.5

def test_human_agreement():
    h_scores = [5.0, 4.0, 3.0, 2.0, 1.0]
    j_scores = [5.0, 4.0, 3.0, 2.0, 1.0]
    stats = HumanJudgeAgreementAnalyzer.compute_agreement(h_scores, j_scores)
    assert stats["quadratic_weighted_kappa"] == 1.0
    assert stats["pearson_correlation"] == 1.0
    assert stats["exact_agreement_rate"] == 1.0
