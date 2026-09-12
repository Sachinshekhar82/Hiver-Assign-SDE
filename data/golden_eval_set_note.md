# Golden Evaluation Dataset: Sampling & Labeling Methodology

## 1. Overview
The Golden Evaluation Dataset comprises **200 hand-curated, stratified examples** built specifically for `@AppleSupport` from the *Customer Support on Twitter* dataset (`thoughtvector/customer-support-on-twitter`). It provides an authoritative ground truth for evaluating:
1. **Intent Classification** across 7 domain-specific support categories.
2. **Escalation Decisions & Stated Reasons** (`AUTO_HANDLE` vs. `ESCALATE`).
3. **Response Drafting Quality** grounded in historical brand resolutions.
4. **Human-Judge Calibration** to empirically measure LLM-as-judge alignment.

---

## 2. Sampling Methodology
The 200 test cases were constructed using a three-stage sampling and refinement process:

1. **Empirical Extraction from Real-World Conversations**:
   - Filtered ~76,639 `@AppleSupport` threads from the Twitter corpus.
   - Identified recurring clusters of customer complaints, device models (iPhone 6s through X, iPad Pro, Apple Watch Series 3, MacBook Pro), and iOS versions (iOS 10/11 transitions).
   - Extracted representative customer problem statements and corresponding official Apple Support responses.

2. **Intent Stratification**:
   - The dataset was stratified across 7 mutually exclusive intents derived from the empirical support distribution:
     - `hardware_battery` (30 cases / 15.0%): battery drain, cold shutdown, bulging battery, charging port damage.
     - `software_os_update` (30 cases / 15.0%): update verification freeze, app crash loops, storage cache, kernel reboot.
     - `account_appleid_icloud` (28 cases / 14.0%): locked ID, lost 2FA codes, account takeover, iCloud sync.
     - `connectivity_network` (28 cases / 14.0%): "No SIM", greyed Wi-Fi, car Bluetooth stutter, cellular drops.
     - `billing_subscriptions` (28 cases / 14.0%): unexpected subscriptions, App Store refunds, duplicate charges.
     - `physical_damage_repair` (28 cases / 14.0%): shattered glass, liquid ingress, quiet ear speaker, AppleCare terms.
     - `general_inquiry_features` (28 cases / 14.0%): screen orientation padlock, deleting apps, Trade In, gestures.

3. **Difficulty & Edge-Case Injection**:
   - **Easy (97 cases / 48.5%)**: Unambiguous phrasing, direct keyword signals ("how do I cancel Apple Music", "padlock icon").
   - **Medium (78 cases / 39.0%)**: Implicit symptoms ("phone turns off in the cold", "yellow screen tint", "storage taking 45GB").
   - **Hard / Adversarial (25 cases / 12.5%)**: Angry/frustrated customers ("I tried 3 cables, it still doesn't charge, replace it now!"), acute physical hazards ("battery bulging pushing screen off", "charger sparkled with burning smell"), active security breaches ("someone in Russia changed my password"), and legal threats.

---

## 3. Ground-Truth Escalation Criteria
Every case was labelled with an explicit binary escalation flag (`true_escalate`) and a domain-grounded `true_escalation_reason`:

- **AUTO_HANDLE (`true_escalate = False`, 142 cases / 71.0%)**:
  - The problem is resolvable via standard client-side troubleshooting (force restart, resetting network settings, toggling settings, updating iOS, deleting app cache, self-service URLs like `reportaproblem.apple.com` or `iforgot.apple.com`).
  - No physical inspection, store booking, financial refund ledger access, or tier-2 security override is required.

- **ESCALATE (`true_escalate = True`, 58 cases / 29.0%)**:
  - **Hardware Failure / Physical Damage**: Shattered front/back glass, liquid immersion, blown speaker, broken SIM tray, greyed-out Wi-Fi IC, unrecoverable bootloops.
  - **Acute Safety Hazards**: Bulging battery, burning electrical smell, sparking chargers (immediate cessation of use and safety escalation).
  - **Account Security & Legal Compromise**: Active account hijacking, deceased account estate requests, legal threats.
  - **Repeated Failure / Frustration**: Customer has already attempted multiple standard fixes without resolution.

---

## 4. Human Calibration Ratings
Each case was assigned an expert `human_judge_score` (1 to 5) evaluating the gold reference reply across clarity, empathy, correctness, and brand voice. This gold scoring distribution serves as the benchmark against which the automated LLM-as-judge rubric is validated for inter-rater agreement (Cohen's Kappa and Pearson correlation).
