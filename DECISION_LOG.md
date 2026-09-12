# Engineering Decision Log (15 Non-Obvious Decisions)

This document details 15 non-obvious technical, architectural, and product decisions made while designing, implementing, and evaluating the `@AppleSupport` AI support system.

---

### Decision 1: Choosing `@AppleSupport` Over `@AmazonHelp` or Airlines
- **Decision**: Selected `@AppleSupport` (76,639 conversations) as the primary brand over `AmazonHelp` (81,092) or airlines (`Delta`, `British_Airways`).
- **Context / Alternatives Considered**: Amazon was slightly larger in raw volume.
- **Why Non-Obvious & Rationale**: Amazon issues almost exclusively require real-time private database queries (tracking IDs, carrier logistics, live delivery trucks). Without mock logistics databases, grounding becomes artificial. Apple Support deals with repeatable, self-contained technical troubleshooting (iOS updates, hardware diagnostics, battery health, settings paths) that can be genuinely verified, grounded, and evaluated offline.

---

### Decision 2: Decoupling Intent Classification from Escalation Routing
- **Decision**: Implemented Intent Classification and Escalation Routing as separate, independent pipeline stages rather than predicting escalation as an intent sub-class.
- **Context / Alternatives Considered**: Unified prompt predicting `(intent, should_escalate)` in one step.
- **Why Non-Obvious & Rationale**: Escalation is orthogonal to intent. A battery issue can be trivial self-service (`Settings > Battery Health`) or an acute safety emergency (bulging battery). Merging them causes combinatorial explosion in the label space and compromises safety guardrail interpretability.

---

### Decision 3: Defining 7 Coarse Technical Intents Instead of Banking77-Style Fine Intents
- **Decision**: Defined 7 cohesive technical support intents rather than 50+ granular micro-intents.
- **Context / Alternatives Considered**: Fine-grained taxonomy (e.g. `battery_health`, `battery_cold`, `battery_drain`, `cable_error`).
- **Why Non-Obvious & Rationale**: On Twitter, customer messages are terse and frequently combine symptoms. Granular taxonomies suffer from massive inter-annotator disagreement and low classifier precision without adding routing value. 7 actionable categories cleanly map to specialized tier-2 resolver groups (Genius Bar, iTunes Billing, Account Security, iOS Triage).

---

### Decision 4: Defining False Auto-Handle Rate (FAHR) as the Primary Safety KPI
- **Decision**: Prioritized False Auto-Handle Rate ($FAHR = FN / (FN + TP)$) over standard accuracy and precision in escalation evaluation.
- **Context / Alternatives Considered**: Relying on overall classification accuracy or F1-score.
- **Why Non-Obvious & Rationale**: In high-imbalance safety routing, an agent with 70% accuracy can have a 100% failure rate on safety hazards. Auto-handling a bulging battery or stolen credentials is unacceptable. FAHR isolates the dangerous failure cases and prevents the accuracy paradox from masking safety risks.

---

### Decision 5: Calibrated Hybrid Architecture Over Pure Zero-Shot LLM API Calls
- **Decision**: Built a hybrid classifier combining calibrated TF-IDF statistical features with domain regex rules and temperature-scaled logits, backed by a local RAG memory.
- **Context / Alternatives Considered**: Calling external LLM APIs (Gemini/OpenAI) for every single classification turn.
- **Why Non-Obvious & Rationale**: External APIs introduce network latency (500–2000 ms), cost scaling bottlenecks, rate limits, and failure points. The hybrid architecture executes in < 15 ms per query, runs 100% deterministically offline, and allows immediate reproduction on any evaluator's machine without requiring paid API keys.

---

### Decision 6: Informational Regex Bypass to Prevent Over-Escalation
- **Decision**: Added an informational regex bypass for repair/hardware inquiries that only ask for pricing or preparation (`how much`, `cost`, `pricing`, `what to bring`).
- **Context / Alternatives Considered**: Escalating any message classified under `physical_damage_repair`.
- **Why Non-Obvious & Rationale**: Many users tweeting about broken screens just want to know the AppleCare+ screen deductible ($29) or Genius Bar appointment prep. Auto-handling informational inquiries protects human tier-2 bandwidth and lowers False Escalation Rate (FER).

---

### Decision 7: Multi-Factor Escalation Tiers with Risk Categories
- **Decision**: Structured escalation into 5 explicit severity categories (`acute_safety`, `hardware_failure`, `security_fraud`, `billing_dispute`, `customer_frustration`) rather than a single boolean flag.
- **Context / Alternatives Considered**: A single binary threshold output.
- **Why Non-Obvious & Rationale**: Different risks require fundamentally different responses and routing. An acute safety hazard requires an immediate command to unplug the device and cessation of use, while a billing dispute requires directing the user to `reportaproblem.apple.com`.

---

### Decision 8: Curating Exactly 200 Golden Evaluation Cases with Stratification
- **Decision**: Hand-crafted and verified exactly 200 stratified evaluation cases rather than using a random train/test split of Kaggle tweets.
- **Context / Alternatives Considered**: Randomly holding out 5% of raw Twitter threads.
- **Why Non-Obvious & Rationale**: Raw Twitter data is heavily polluted with bot spam, broken links, single-word replies ("ok", "thanks"), and incomplete conversational fragments. Random splits test data cleaning artifacts, not agent intelligence. A hand-curated golden set provides clean ground truth for intent, escalation, and reference replies.

---

### Decision 9: Pre-Filtering Historical Twitter URLs and Handles
- **Decision**: Preprocessed all raw customer support data to strip ephemeral `t.co` shortlinks and customer handles.
- **Context / Alternatives Considered**: Keeping raw tweet tokens.
- **Why Non-Obvious & Rationale**: Twitter shortlinks in 2017 dataset (`https://t.co/...`) are now dead 404 links. Allowing a model to generate dead shortlinks would degrade real-world response quality. We replaced them with canonical Apple URLs (`support.apple.com/repair`, `iforgot.apple.com`).

---

### Decision 10: Human Calibration Study on a Balanced Quality Distribution
- **Decision**: Evaluated human-judge agreement on an intentional 50-pair calibration set spanning the entire 1 to 5 star spectrum rather than only high-scoring responses.
- **Context / Alternatives Considered**: Computing correlation solely on the Golden set reference replies.
- **Why Non-Obvious & Rationale**: Golden reference responses are all high quality (4–5 stars). In statistics, evaluating correlation on a near-constant variable yields degenerate near-zero variance. Testing across bad, mediocre, and excellent replies proved the judge exhibits strong correlation ($r = 0.919$) across the full quality spectrum.

---

### Decision 11: Quadratic Weighted Kappa Instead of Linear Kappa
- **Decision**: Used Cohen's Quadratic Weighted Kappa ($\kappa_w$) rather than unweighted Cohen's Kappa for human-judge agreement.
- **Context / Alternatives Considered**: Standard unweighted Cohen's Kappa or exact accuracy.
- **Why Non-Obvious & Rationale**: Unweighted Kappa treats a minor 4 vs. 5 disagreement the same as a catastrophic 1 vs. 5 disagreement. Quadratic weighting properly penalizes large disagreements while recognizing near-consensus on ordinal Likert scales.

---

### Decision 12: Persona Grounding: Mandating DM Inquiries on Escalation
- **Decision**: Programmed the response generator to strictly append Direct Message (DM) invitations and ask for device serial numbers privately whenever escalating.
- **Context / Alternatives Considered**: Giving general advice and ending the turn.
- **Why Non-Obvious & Rationale**: Real Apple Support Twitter protocol never resolves hardware or account disputes in public tweets. A response that fails to transition the customer to a secure private channel is operationally deficient.

---

### Decision 13: Offline-First Evaluation Pipeline with Zero External Dependencies
- **Decision**: Engineered the pipeline so that `python run_pipeline.py` executes out-of-the-box in under 15 seconds without requiring an API key, while still supporting Gemini/OpenAI API keys if passed via environment variables.
- **Context / Alternatives Considered**: Requiring evaluators to input a paid API key before running.
- **Why Non-Obvious & Rationale**: The assignment instructions explicitly state: *"README must let us reproduce your headline results in under 15 minutes."* An API key requirement causes instant friction, quota limits, and irreproducibility. Offline-first guarantees frictionless evaluation.

---

### Decision 14: Intent Fallback Templates Over Monolithic Hallucination
- **Decision**: Implemented intent-scoped fallback templates when historical nearest-neighbor similarity falls below threshold ($< 0.35$).
- **Context / Alternatives Considered**: Forcing retrieval to pick the closest tweet even if cosine similarity is 0.05.
- **Why Non-Obvious & Rationale**: Low-similarity retrieval returns irrelevant resolutions (e.g. offering watchOS advice for an iPhone problem). Intent-guided templates guarantee safe, accurate troubleshooting paths when semantic retrieval confidence is low.

---

### Decision 15: Single CLI Command for End-to-End Evaluation
- **Decision**: Unified all baselines, metrics, generation scoring, and calibration agreement into a single root script: `python run_pipeline.py`.
- **Context / Alternatives Considered**: Splitting evaluation into 4 separate scripts (`eval_intent.py`, `eval_escalate.py`, `eval_gen.py`, `eval_judge.py`).
- **Why Non-Obvious & Rationale**: Fragmented scripts increase reviewer overhead and risk execution sequence errors. A single unified entry point provides immediate transparency and reproduces all headline figures in one unified terminal view.
