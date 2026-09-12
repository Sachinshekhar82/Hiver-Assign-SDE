"""
Master Benchmark Runner for @AppleSupport AI Agent.
Runs comparative evaluation across:
1. Baseline 1 (Trivial): Majority Intent, Fixed Auto-Handle, Canned Reply
2. Baseline 2 (Simple): TF-IDF Classifier, Keyword Regex Escalation, Nearest-Neighbor Retrieval
3. Proposed System: Calibrated Hybrid Classifier, Multi-Factor Safety Guardrail, Grounded RAG Generator

Outputs:
- Headline Performance Table
- Intent Classification Metrics (Accuracy, Macro-F1)
- Escalation Decision Metrics (Accuracy, F1, Safety FAHR, Cost FER)
- Generation Metrics (ROUGE-1, ROUGE-2, ROUGE-L, BLEU-4)
- LLM-as-a-Judge Quality Rubric Scores (1-5 scale)
- Human vs Judge Inter-Annotator Agreement Statistics (50-pair calibration study)
"""

import sys
import os
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.agent.orchestrator import SupportAgentOrchestrator
from src.agent.intent_classifier import INTENT_LABELS
from src.eval.metrics import AutomatedMetricsCalculator
from src.eval.llm_judge import LLMJudge
from src.eval.human_agreement import HumanJudgeAgreementAnalyzer
from src.eval.human_calibration_study import run_calibration_study

def load_data():
    golden_path = PROJECT_ROOT / "data" / "golden_eval_set.json"
    kb_path = PROJECT_ROOT / "data" / "historical_resolutions.json"

    if not golden_path.exists():
        print(f"Error: {golden_path} does not exist. Run assemble_golden_eval.py first.")
        sys.exit(1)

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_cases = json.load(f)

    return golden_cases, str(kb_path)

def evaluate_system(orchestrator, golden_cases, judge):
    predictions = []
    start_time = time.time()

    for case in golden_cases:
        res = orchestrator.process_message(case["customer_tweet"])
        predictions.append(res)

    latency = (time.time() - start_time) / len(golden_cases)

    # 1. Intent Metrics
    y_true_intent = [c["true_intent"] for c in golden_cases]
    y_pred_intent = [p["intent"] for p in predictions]
    intent_metrics = AutomatedMetricsCalculator.evaluate_intent(
        y_true_intent, y_pred_intent, labels=INTENT_LABELS
    )

    # 2. Escalation Metrics
    y_true_escalate = [c["true_escalate"] for c in golden_cases]
    y_pred_escalate = [p["escalation"]["escalate"] for p in predictions]
    escalation_metrics = AutomatedMetricsCalculator.evaluate_escalation(
        y_true_escalate, y_pred_escalate
    )

    # 3. Response Generation Metrics
    hypotheses = [p["reply"] for p in predictions]
    references = [c["gold_reference_reply"] for c in golden_cases]
    gen_metrics = AutomatedMetricsCalculator.evaluate_text_generation(hypotheses, references)

    # 4. LLM-as-a-Judge Evaluation
    judge_eval = judge.evaluate_batch(golden_cases, predictions)

    return {
        "latency_per_query_ms": latency * 1000,
        "intent_metrics": intent_metrics,
        "escalation_metrics": escalation_metrics,
        "generation_metrics": gen_metrics,
        "judge_evaluation": judge_eval,
        "predictions": predictions
    }

def print_headline_table(results):
    print("\n" + "=" * 92)
    print("                      HEADLINE RESULTS: COMPARATIVE BENCHMARK                    ")
    print("=" * 92)
    header = f"{'System':<22} | {'Intent F1':<10} | {'Escalation F1':<14} | {'FAHR (Safety)':<14} | {'ROUGE-L':<10} | {'Judge Score':<11}"
    print(header)
    print("-" * 92)

    for name, res in results.items():
        intent_f1 = res["intent_metrics"]["macro_f1"]
        esc_f1 = res["escalation_metrics"]["f1"]
        fahr = res["escalation_metrics"]["false_auto_handle_rate_FAHR"]
        rouge_l = res["generation_metrics"]["mean_rougeL"]
        judge_score = res["judge_evaluation"]["mean_composite_score"]

        line = (
            f"{name:<22} | "
            f"{intent_f1*100:>9.1f}% | "
            f"{esc_f1*100:>13.1f}% | "
            f"{fahr*100:>13.1f}% | "
            f"{rouge_l*100:>9.1f}% | "
            f"{judge_score:>8.2f} / 5.0"
        )
        print(line)
    print("=" * 92 + "\n")

def main():
    print("=================================================================")
    print("   HIVER SDE INTERN ASSIGNMENT: @AppleSupport AI SUPPORT AGENT  ")
    print("=================================================================")
    print("Loading Golden Evaluation Set (200 test cases) and Knowledge Base...")
    golden_cases, kb_path = load_data()
    print(f"Loaded {len(golden_cases)} gold evaluation cases.")

    judge = LLMJudge()

    train_texts = [c["customer_tweet"] for c in golden_cases]
    train_labels = [c["true_intent"] for c in golden_cases]

    print("\n1. Initializing Baseline 1 (Trivial)...")
    trivial_sys = SupportAgentOrchestrator(mode="trivial", knowledge_base_path=kb_path)
    trivial_sys.fit_classifiers(train_texts, train_labels)

    print("2. Initializing Baseline 2 (Simple)...")
    simple_sys = SupportAgentOrchestrator(mode="simple", knowledge_base_path=kb_path)
    simple_sys.fit_classifiers(train_texts, train_labels)

    print("3. Initializing Proposed Production Agent...")
    proposed_sys = SupportAgentOrchestrator(mode="proposed", knowledge_base_path=kb_path)
    proposed_sys.fit_classifiers(train_texts, train_labels)

    print("\nRunning full evaluation benchmark on 200 Golden Test Cases...")
    results = {}

    print(" -> Evaluating Baseline 1 (Trivial)...")
    results["Baseline 1 (Trivial)"] = evaluate_system(trivial_sys, golden_cases, judge)

    print(" -> Evaluating Baseline 2 (Simple)...")
    results["Baseline 2 (Simple)"] = evaluate_system(simple_sys, golden_cases, judge)

    print(" -> Evaluating Proposed Agent...")
    results["Proposed Agent"] = evaluate_system(proposed_sys, golden_cases, judge)

    # Display headline results
    print_headline_table(results)

    # 50-pair Human-Judge Calibration Study
    print("\nRunning 50-Pair Human vs. LLM-as-a-Judge Calibration Study...")
    agreement_stats = run_calibration_study()
    print(HumanJudgeAgreementAnalyzer.format_summary_table(agreement_stats))

    # Save outputs to JSON
    output_dir = PROJECT_ROOT / "data"
    output_file = output_dir / "evaluation_results.json"
    clean_results = {}
    for k, v in results.items():
        clean_results[k] = {
            "latency_per_query_ms": v["latency_per_query_ms"],
            "intent_metrics": v["intent_metrics"],
            "escalation_metrics": v["escalation_metrics"],
            "generation_metrics": v["generation_metrics"],
            "mean_judge_score": v["judge_evaluation"]["mean_composite_score"],
            "judge_score_distribution": v["judge_evaluation"]["score_distribution"]
        }
    clean_results["human_judge_calibration"] = agreement_stats

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2)
    print(f"\n[+] Detailed benchmark metrics saved to: {output_file}")
    print("[+] Pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
