# Intain-Sight: 5-Minute Demo Script

> **Objective:** Strictly timed presentation targeting the 15 required Intain benchmark flow points.

---

### [0:00 - 0:30] Introduction & Pool Profiling
*Open the Streamlit App to **1. Pool Overview**.*
**Speaker:** "Welcome to Intain-Sight. We’ve built an enterprise verification agent that shifts securitization analysis from manual auditing to predictive intelligence. Here, you see our Top-Level Pool Profiling. Our Polars-backed engine instantly calculates aggregate health metrics like the exact Weighted Average Coupon (WAC) and a proprietary 12-month projected Conditional Default Rate."

### [0:30 - 1:15] ML Architecture & Leakage Prevention
*Point to the Calibration curves and switch to **2. Loan Explorer**.*
**Speaker:** "The foundation of these probabilities is our Multi-Task GBDT Ensemble. Crucially, as per compliance mandates, our training pipeline enforces a strict 12-month blackout split, ensuring zero temporal data leakage. We use Focal Loss to handle severe class imbalance, allowing us to predict competing risks—Default versus Prepayment—with high precision."

### [1:15 - 2:00] Loan-Level Explainability (SHAP & Counterfactuals)
*Select a high-risk loan in **Loan Explorer**.*
**Speaker:** "Black-box AI is unacceptable in underwriting. For every loan, we generate Local SHAP attributions, identifying precisely which features—like a high DTI or dropping credit score—drive the risk. But we don't just stop at why. Notice our **Counterfactual Prescription** engine: it performs a greedy stochastic search to recommend actionable, business-realistic concessions (like a balance paydown) required to flip this loan from 'Reject' back to 'Approved'."

### [2:00 - 2:45] Survival Modeling & Anomaly Rules
*Switch to **4. Verification Queue**.*
**Speaker:** "Behind the scenes, we merge continuous survival curves—built using the Aalen-Johansen estimator—with deterministic accounting rules. The result is a hybrid anomaly score. When a loan fails a hard rule—like a servicer balance mismatch—it lands directly in this Verification Queue for triage."

### [2:45 - 3:30] Macro Scenario Sandbox
*Switch to **3. Scenario Sandbox**.*
**Speaker:** "Let’s look at macro exposure. Using our Monte Carlo simulation engine, underwriters can shock the portfolio. By adjusting Interest Rates and Unemployment, the engine maps these macro scalars to micro-loan features, running 1,000 vectorized iterations in under 500ms to produce 90% Confidence Interval stress curves."

### [3:30 - 4:15] The Governed LLM Copilot
*Switch to **5. LLM Copilot**.*
**Speaker:** "To assist analysts in the queue, we integrated a RAG-backed LLM Copilot. It retrieves exact definitions from our data dictionaries to ground its logic. More importantly, it is **strictly governed**."

### [4:15 - 5:00] Hallucination Guardrails & Audit Logging
*Point to the **Live Compliance Ledger**.*
**Speaker:** "If the LLM hallucinates a probability—for instance, claiming a 45% default rate when our ML model calculated 18%—our independent numerical guardrail intercepts and rejects the response instantly. Every interaction, passed or rejected, is immutably written to this SQLite audit log. 

Ultimately, Intain-Sight compiles all this intelligence into a flawless, zero-null `submission.csv` contract. Thank you."
