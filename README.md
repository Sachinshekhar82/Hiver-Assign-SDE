# Apple Support AI Agent & Evaluation System (@AppleSupport)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](tests/)
[![Reproducibility](https://img.shields.io/badge/reproducibility-%3C%2015%20seconds-success.svg)](run_pipeline.py)
[![Brand](https://img.shields.io/badge/brand-%40AppleSupport-black.svg)](https://twitter.com/AppleSupport)

An end-to-end, production-grade AI Customer Support Agent and Evaluation Harness engineered for **`@AppleSupport`** using real-world customer support conversations on Twitter. Built for the **Hiver SDE Intern Take-Home Assignment**.

---

## Quickstart: Reproduce Headline Results in Under 15 Seconds

You can reproduce all headline metrics right out of the box with zero external dependencies and zero required API keys.

```bash
# 1. Clone the repository
git clone https://github.com/Sachinshekhar82/Hiver-Assign-SDE.git
cd Hiver-Assign-SDE

# 2. Install dependencies (scikit-learn, pandas, polars, scipy, pytest)
pip install -r requirements.txt

# 3. Run the complete master evaluation benchmark
python run_pipeline.py
```

### Run Unit Tests
```bash
python -m pytest tests/
```

---

## Headline Results: Comparative Benchmark

Evaluated against the hand-labelled **Golden Evaluation Set of 200 authentic customer cases**:

| System | Intent Macro-F1 | Escalation F1 | FAHR (Safety Risk) | ROUGE-L | LLM Judge Score (1-5) | Latency / Query |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 (Trivial)** | 3.7% | 0.0% | 100.0% | 13.0% | 3.84 / 5.0 | < 1 ms |
| **Baseline 2 (Simple)** | **99.5%** | 27.4% | 82.8% | 9.2% | 3.32 / 5.0 | 4.2 ms |
| **Proposed Production Agent** | 89.5% | **66.0%** | **43.1%** | **15.4%** | **4.13 / 5.0** | 12.8 ms |

> [!IMPORTANT]
> **Key Safety Breakthrough**: The Proposed Agent cuts the **False Auto-Handle Rate (FAHR)** by **39.7 percentage points** compared to the simple baseline, preventing critical battery hazards, hardware failures, and account takeovers from being erroneously automated.

### Human vs. LLM-as-a-Judge Agreement (50-Pair Calibration Study)
To prove our LLM Judge can be trusted, we conducted a 50-pair human calibration study across the full quality spectrum (1 to 5 stars):
- **Cohen's Quadratic Weighted Kappa ($\kappa_w$)**: `0.6000` (Substantial inter-rater agreement)
- **Pearson Linear Correlation ($r$)**: `0.9192` ($p = 4.70 \times 10^{-21}$)
- **Spearman Rank Correlation ($\rho$)**: `0.9161` ($p = 1.11 \times 10^{-20}$)
- **Mean Absolute Error (MAE)**: `0.8900 stars`
- **Within-1-Point Agreement Rate**: `82.0%`

---

## Deliverables & Documentation Index

Every deliverable requested in the assignment is thoroughly documented and version-controlled:

1. **[Technical Evaluation Report](REPORT.md)** (6-page equivalent)
   - Problem Framing: What "good" means for `@AppleSupport`, and what we chose *not* to build.
   - Comparative Results vs. Dual Baselines (Trivial and Simple).
   - Top 5 Failure Modes with real customer examples, root causes, and architectural mitigations.
   - **Mandatory Section**: *"What is misleading about my headline number?"*
   - Roadmap: What we would build next with one more week.
2. **[Engineering Decision Log](DECISION_LOG.md)**
   - Plain list of the 15 non-obvious architectural, machine learning, and product decisions made and why.
3. **[Golden Evaluation Dataset Note](data/golden_eval_set_note.md)**
   - Sampling methodology from 76,639 AppleSupport threads.
   - Intent taxonomy definitions, escalation criteria, difficulty stratification, and human calibration ratings.
4. **Golden Evaluation Set Data Files**:
   - [`data/golden_eval_set.json`](data/golden_eval_set.json) (200 curated cases in JSON)
   - [`data/golden_eval_set.csv`](data/golden_eval_set.csv) (200 curated cases in tabular CSV)
5. **Historical Resolution Knowledge Base**:
   - [`data/historical_resolutions.json`](data/historical_resolutions.json) (1,500 curated resolution pairs)

---

## System Architecture

```
                                 Customer Tweet
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │  1. Preprocessor & Entities   │
                       │  (Clean handles, URLs, model) │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │  2. Intent Classifier         │
                       │  (Calibrated Hybrid Model)    │
                       └───────┬───────────────┬───────┘
                               │               │
                               ▼               ▼
           ┌──────────────────────────────┐  ┌──────────────────────────────┐
           │ 3. Escalation Guardrail      │  │ 4. Grounded RAG Generator    │
           │ - Critical Safety Hazards    │  │ - Top-3 Exemplar Retrieval   │
           │ - Hardware Physical Damage   │  │ - Apple Persona Alignment    │
           │ - Account Security / Fraud   │  │ - Safety Routing & Links     │
           │ - Customer Frustration/SLA   │  │ - Step-by-Step Guidance      │
           └──────────────┬───────────────┘  └──────────────┬───────────────┘
                          │                                 │
                          └────────────────┬────────────────┘
                                           ▼
                       ┌───────────────────────────────┐
                       │    Structured Support Output  │
                       │    - Intent & Confidence      │
                       │    - Escalation & Stated Reason│
                       │    - Drafted Grounded Reply   │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │  5. Evaluation Harness        │
                       │  - Automated Accuracy/F1/FAHR │
                       │  - ROUGE-1/2/L & BLEU-4       │
                       │  - LLM-as-a-Judge Rubric      │
                       │  - Human-Judge Calibration    │
                       └───────────────────────────────┘
```

---

## Repository Structure

```
.
├── README.md                          # Quickstart, headline results, architecture
├── REPORT.md                          # Comprehensive 6-page technical report
├── DECISION_LOG.md                    # 15 non-obvious engineering decisions
├── requirements.txt                   # Minimal Python dependencies
├── run_pipeline.py                    # Master benchmark runner (< 15 seconds)
├── data/
│   ├── golden_eval_set.json           # 200 hand-labelled evaluation cases (JSON)
│   ├── golden_eval_set.csv            # 200 hand-labelled evaluation cases (CSV)
│   ├── golden_eval_set_note.md        # Note on sampling & labeling methodology
│   ├── historical_resolutions.json    # 1,500 curated AppleSupport resolution pairs
│   └── evaluation_results.json        # Serialized benchmark metrics & distributions
├── src/
│   ├── data/
│   │   ├── preprocessor.py            # Tweet cleaner & entity extractor
│   │   ├── build_knowledge_base.py    # Extracts resolution pairs from raw corpus
│   │   ├── assemble_golden_eval.py    # Assembles the 200 golden cases
│   │   └── intents_data/              # Intent dataset generation modules
│   ├── agent/
│   │   ├── intent_classifier.py       # Trivial, Simple, and Calibrated Hybrid Classifiers
│   │   ├── escalation_engine.py       # Trivial, Simple, and Multi-Factor Safety Engines
│   │   ├── response_generator.py      # Canned, Nearest-Neighbor, and Grounded RAG Generators
│   │   └── orchestrator.py            # Unified pipeline coordinator
│   └── eval/
│       ├── metrics.py                 # Intent F1, Escalation F1, FAHR, FER, ROUGE, BLEU
│       ├── llm_judge.py               # 5-criteria LLM-as-a-Judge rubric engine
│       ├── human_agreement.py         # Cohen's Kappa, Pearson r, Spearman rho, MAE
│       └── human_calibration_study.py # 50-pair human agreement calibration study
└── tests/
    └── test_pipeline.py               # Unit & integration test suite
```

---

## Citations & Dataset Attribution

In compliance with the assignment rules:
- **Primary Dataset**: Thoughtvector's *Customer Support on Twitter* corpus, Kaggle / Hugging Face mirror (`TNE-AI/customer-support-on-twitter-conversation`).
- **Scikit-Learn**: Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, pp. 2825-2830, 2011.
- **Evaluation Rubric**: Designed following Anthropic and OpenAI LLM-as-a-Judge methodologies with Cohen's Quadratic Weighted Kappa calibration.
