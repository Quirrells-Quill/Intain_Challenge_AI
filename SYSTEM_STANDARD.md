# INTAIN-SIGHT: SYSTEM ARCHITECTURAL STANDARD
> Version: 1.0 | Status: BINDING | Applied to all Stages 0–N

---

## PILLAR 1 — SUBMISSION CONTRACT & SCHEMA IMMUTABILITY

Every inference pipeline must output a submission_template.csv-compatible schema 
with ZERO missing values, exact column naming, and matching row order.

| # | Column                | Type             | Constraints                                                     |
|---|-----------------------|------------------|-----------------------------------------------------------------|
| 1 | loan_id               | String / Int     | Matches test set identifier                                     |
| 2 | reporting_month       | Timestamp / Date | Evaluation record date                                          |
| 3 | prob_3m_delinq        | Float            | [0.0, 1.0] calibrated probability                               |
| 4 | prob_6m_delinq        | Float            | [0.0, 1.0] calibrated probability                               |
| 5 | prob_12m_default      | Float            | [0.0, 1.0] calibrated probability                               |
| 6 | prob_12m_prepay       | Float            | [0.0, 1.0] calibrated probability                               |
| 7 | predicted_next_state  | String           | Current, Delinquent, Default, Prepaid                           |
| 8 | exception_required    | Boolean          | True if anomaly_score > threshold OR rule_violation > 0         |
| 9 | exception_type        | String           | Data Logic Error / Servicer Discrepancy / Severe Deterioration / None |
|10 | anomaly_score         | Float            | Normalized [0.0, 100.0]                                         |
|11 | top_drivers           | String           | Semicolon-delimited top 3 SHAP/Rule driver codes                |
|12 | recommended_action    | String           | Auto-Approve / Manual Triage / Reject/Repurchase                |
|13 | confidence            | Float            | [0.0, 1.0] derived from ensemble variance                       |

DISQUALIFICATION RISK: Schema deviations cause automated harness failure.

---

## PILLAR 2 — DYNAMIC INGESTION (NO HARDCODING)

- All domain thresholds loaded from configs/validation_rules.json via ValidationRuleEngine.
- All field types, categorical limits, semantic descriptions loaded from data/data_dictionary.md.
- Thresholds are NEVER hardcoded in Python source files.

---

## PILLAR 3 — TEMPORAL ZERO-LEAKAGE ENFORCEMENT

Train Period : [T_start            ->  T_train_cutoff]
Blackout     : (T_train_cutoff     ->  T_train_cutoff + 12 months]  <- DROPPED
OOT Period   : [T_train_cutoff + 12 months  ->  T_eval_end]

- K-Fold and random splits are STRICTLY FORBIDDEN.
- Any loan_id in OOT must not expose future performance signals into training features.

---

## PILLAR 4 — DUAL-TRACK PREDICTION ARCHITECTURE

Benchmark Engine : LightGBM + CatBoost + XGBoost (Cost-sensitive, Isotonic calibration)
Novelty Engine   : PyTorch MTL Network (Shared backbone + 5 task heads)

Both tracks log real-time metrics to MLflow for side-by-side comparison.

---

## PILLAR 5 — SECURITIZATION POOL-LEVEL INTELLIGENCE

Pool-level aggregates computed for every scoring batch:
- WAC  -- Weighted Average Coupon
- WAM  -- Weighted Average Maturity
- CDR  -- Conditional Default Rate
- CPR  -- Conditional Prepayment Rate
- Pool Health Score (0-100 composite index)
- Geographic and credit band concentration risks
