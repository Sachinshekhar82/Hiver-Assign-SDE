"""
Support Agent Orchestrator for @AppleSupport.
Coordinates preprocessing, intent classification, escalation policy, and response generation.
"""

from typing import Dict, Any, Optional
from src.data.preprocessor import TweetPreprocessor
from src.agent.intent_classifier import (
    MajorityClassBaseline,
    TfidfBaseline,
    CalibratedHybridClassifier
)
from src.agent.escalation_engine import (
    TrivialEscalationBaseline,
    SimpleRegexEscalationBaseline,
    MultiFactorEscalationGuardrail
)
from src.agent.response_generator import (
    TrivialResponseBaseline,
    SimpleRetrievalBaseline,
    GroundedRAGResponseGenerator
)

class SupportAgentOrchestrator:
    def __init__(self, mode: str = "proposed", knowledge_base_path: Optional[str] = None):
        self.mode = mode
        self.preprocessor = TweetPreprocessor()

        if mode == "trivial":
            self.intent_clf = MajorityClassBaseline()
            self.escalation_engine = TrivialEscalationBaseline()
            self.response_gen = TrivialResponseBaseline()
        elif mode == "simple":
            self.intent_clf = TfidfBaseline()
            self.escalation_engine = SimpleRegexEscalationBaseline()
            self.response_gen = SimpleRetrievalBaseline(knowledge_base_path=knowledge_base_path)
        else: # "proposed"
            self.intent_clf = CalibratedHybridClassifier()
            self.escalation_engine = MultiFactorEscalationGuardrail()
            self.response_gen = GroundedRAGResponseGenerator(knowledge_base_path=knowledge_base_path)

    def fit_classifiers(self, texts, labels):
        if hasattr(self.intent_clf, "fit"):
            self.intent_clf.fit(texts, labels)

    def process_message(self, tweet_text: str) -> Dict[str, Any]:
        preprocessed = self.preprocessor.preprocess(tweet_text)
        cleaned = preprocessed["cleaned_text"]
        entities = preprocessed["entities"]

        # 1. Intent Classification
        if isinstance(self.intent_clf, CalibratedHybridClassifier):
            intent, intent_conf = self.intent_clf.predict(cleaned)
        elif isinstance(self.intent_clf, TfidfBaseline):
            intent = self.intent_clf.predict(cleaned)
            intent_conf = max(self.intent_clf.predict_proba(cleaned).values())
        else:
            intent = self.intent_clf.predict(cleaned)
            intent_conf = 0.5

        # 2. Escalation Decision & Stated Reason
        escalation = self.escalation_engine.evaluate(cleaned, intent=intent)

        # 3. Grounded Response Generation
        if self.mode == "proposed":
            reply = self.response_gen.generate(cleaned, intent=intent, escalation=escalation, entities=entities)
        elif self.mode == "simple":
            reply = self.response_gen.generate(cleaned, intent=intent, escalation=escalation)
        else:
            reply = self.response_gen.generate(cleaned)

        return {
            "customer_text": tweet_text,
            "cleaned_text": cleaned,
            "entities": entities,
            "intent": intent,
            "intent_confidence": float(intent_conf),
            "escalation": escalation,
            "reply": reply,
            "system_mode": self.mode
        }
