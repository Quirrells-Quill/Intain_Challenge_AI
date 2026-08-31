# Intain-Sight: LLM Failure Analysis & Guardrail Efficacy Report
> Mandatory Deliverable: Documentation of historical LLM failures caught by the governance subsystem.

## Case 1: Numerical Extrapolation Hallucination
**Input Context**: `actual_default_prob = 18.2%`
**Raw LLM Output**: "Borrower exhibits a 45.0% default probability due to market trends."
**Failure Reason**: LLM hallucinated a 45% default probability when the calibrated GBDT predicted 18.2%.
**Detection Mechanism**: Caught by numerical guardrail (Assertion 1: Probability Delta > 0.03).
**Corrective Action**: Intercepted and flagged as `REJECTED_HALLUCINATION`. Replaced with deterministic template: *"DETERMINISTIC FALLBACK: Loan has a default probability of 18.2% and an anomaly score of 10."*

## Case 2: Regulatory Rule Fabrication
**Input Context**: ML Anomalies `['RULE_1', 'RULE_2']`
**Raw LLM Output**: "Loan violates RULE_7 (State Lending Cap)."
**Failure Reason**: LLM suggested a non-existent state lending cap (RULE_7).
**Detection Mechanism**: Caught by RAG dictionary verification (Assertion 2: Claimed rule not in ML validation matrix).
**Corrective Action**: Intercepted and flagged as `REJECTED_HALLUCINATION`. Replaced with deterministic template: *"WARNING: LLM output rejected due to hallucination (Fabricated rule violation: RULE_7). Proceed with manual review."*

## Case 3: Overconfident Approval Recommendation
**Input Context**: `actual_default_prob = 1.0%`, `anomaly_score = 95` (Servicer Balance Conflict)
**Raw LLM Output**: Action: `Auto-Approve`, Notes: "Proceed to fund."
**Failure Reason**: LLM recommended 'Auto-Approve' on a record with a severe servicer balance conflict.
**Detection Mechanism**: Caught and overridden by Tier 2 deterministic reconciliation check.
**Corrective Action**: Overridden LLM action to `Manual Triage`. Appended note: *"TIER 2 OVERRIDE: Severe anomaly score prevents Auto-Approve."*
