# INTAIN-SIGHT: ANOMALY REVIEWER DOSSIER
> Generated: 2026-08-31 04:18:29  |  Total Cases: 20  |  Reviewer Sign-Off Required

---

## Case 01 🔴 — Loan `DOSSIER_018`

| Field                | Value |
|----------------------|-------|
| **Loan ID**          | `DOSSIER_018` |
| **Reporting Month**  | 2023-06-01 |
| **State**            | CA |
| **Vintage**          | 2019 |
| **Servicer**         | Beta Capital |
| **Anomaly Score**    | **93.3 / 100** |
| **Exception Type**   | Severe Deterioration |
| **Recommended Action** | **Reject/Repurchase** |
| **Top Drivers**      | `CurrentLtv;DaysPastDue;InterestRate` |
| **Days Past Due**    | 6 |
| **Current Balance**  | 148484.3622076988 |

**Audit Narrative:** Loan DOSSIER_018 (CA) identified as a multivariate ML outlier (score: 93.3/100, action: 'Reject/Repurchase'). The unsupervised ensemble detected abnormal patterns in: CurrentLtv, DaysPastDue, InterestRate. Days past due: 6. This loan's risk profile deviates significantly from the training population distribution — manual underwriting review recommended.

---

## Case 02 🔴 — Loan `DOSSIER_000`

| Field                | Value |
|----------------------|-------|
| **Loan ID**          | `DOSSIER_000` |
| **Reporting Month**  | 2023-06-01 |
| **State**            | TX |
| **Vintage**          | 2020 |
| **Servicer**         | Alpha Servicing |
| **Anomaly Score**    | **93.2 / 100** |
| **Exception Type**   | Data Logic Error |
| **Recommended Action** | **Reject/Repurchase** |
| **Top Drivers**      | `CurrentLtv;DaysPastDue;InterestRate` |
| **Days Past Due**    | 103 |
| **Current Balance**  | 294078.8101707774 |

**Audit Narrative:** Loan DOSSIER_000 (TX) flagged for 'Reject/Repurchase' with anomaly score 93.2/100 due to deterministic accounting violations: CurrentLtv, DaysPastDue, InterestRate. Current balance reported as $294,078.81 against an original balance of $400,000.00, violating balance integrity constraints. Immediate data correction and servicer confirmation required before pool reporting.

---

## Case 03 🔴 — Loan `DOSSIER_011`

| Field                | Value |
|----------------------|-------|
| **Loan ID**          | `DOSSIER_011` |
| **Reporting Month**  | 2023-06-01 |
| **State**            | TX |
| **Vintage**          | 2022 |
| **Servicer**         | Beta Capital |
| **Anomaly Score**    | **92.5 / 100** |
| **Exception Type**   | Severe Deterioration |
| **Recommended Action** | **Reject/Repurchase** |
| **Top Drivers**      | `CurrentLtv;DaysPastDue;InterestRate` |
| **Days Past Due**    | 46 |
| **Current Balance**  | 150914.83637558544 |

**Audit Narrative:** Loan DOSSIER_011 (TX) identified as a multivariate ML outlier (score: 92.5/100, action: 'Reject/Repurchase'). The unsupervised ensemble detected abnormal patterns in: CurrentLtv, DaysPastDue, InterestRate. Days past due: 46. This loan's risk profile deviates significantly from the training population distribution — manual underwriting review recommended.

---

## Case 04 🔴 — Loan `DOSSIER_013`

| Field                | Value |
|----------------------|-------|
| **Loan ID**          | `DOSSIER_013` |
| **Reporting Month**  | 2023-06-01 |
| **State**            | FL |
| **Vintage**          | 2021 |
| **Servicer**         | Alpha Servicing |
| **Anomaly Score**    | **90.7 / 100** |
| **Exception Type**   | Severe Deterioration |
| **Recommended Action** | **Reject/Repurchase** |
| **Top Drivers**      | `CurrentLtv;DaysPastDue;InterestRate` |
| **Days Past Due**    | 84 |
| **Current Balance**  | 323255.16085768875 |

**Audit Narrative:** Loan DOSSIER_013 (FL) identified as a multivariate ML outlier (score: 90.7/100, action: 'Reject/Repurchase'). The unsupervised ensemble detected abnormal patterns in: CurrentLtv, DaysPastDue, InterestRate. Days past due: 84. This loan's risk profile deviates significantly from the training population distribution — manual underwriting review recommended.

---

## Case 05 🔴 — Loan `DOSSIER_003`

| Field                | Value |
|----------------------|-------|
| **Loan ID**          | `DOSSIER_003` |
| **Reporting Month**  | 2023-06-01 |
| **State**            | TX |
| **Vintage**          | 2022 |
| **Servicer**         | Beta Capital |
| **Anomaly Score**    | **89.3 / 100** |
| **Exception Type**   | Data Logic Error |
| **Recommended Action** | **Reject/Repurchase** |
| **Top Drivers**      | `CurrentLtv;DaysPastDue;InterestRate` |
| **Days Past Due**    | 53 |
| **Current Balance**  | 129533.55262467191 |

**Audit Narrative:** Loan DOSSIER_003 (TX) flagged for 'Reject/Repurchase' with anomaly score 89.3/100 due to deterministic accounting violations: CurrentLtv, DaysPastDue, InterestRate. Current balance reported as $129,533.55 against an original balance of $400,000.00, violating balance integrity constraints. Immediate data correction and servicer confirmation required before pool reporting.

---

## Case 06 🔴 — Loan `DOSSIER_004`

| Field                | Value |
|----------------------|-------|
| **Loan ID**          | `DOSSIER_004` |
| **Reporting Month**  | 2023-06-01 |
| **State**            | FL |
| **Vintage**          | 2018 |
| **Servicer**         | Beta Capital |
| **Anomaly Score**    | **86.6 / 100** |
| **Exception Type**   | Data Logic Error |
| **Recommended Action** | **Reject/Repurchase** |
| **Top Drivers**      | `CurrentLtv;DaysPastDue;InterestRate` |
| **Days Past Due**    | 99 |
| **Current Balance**  | 271082.5396927227 |

**Audit Narrative:** Loan DOSSIER_004 (FL) flagged for 'Reject/Repurchase' with anomaly score 86.6/100 due to deterministic accounting violations: CurrentLtv, DaysPastDue, InterestRate. Current balance reported as $271,082.54 against an original balance of $400,000.00, violating balance integrity constraints. Immediate data correction and servicer confirmation required before pool reporting.

---

## Case 07 🔴 — Loan `DOSSIER_012`

| Field                | Value |
|----------------------|-------|
| **Loan ID**          | `DOSSIER_012` |
| **Reporting Month**  | 2023-06-01 |
| **State**            | TX |
| **Vintage**          | 2018 |
| **Servicer**         | Beta Capital |
| **Anomaly Score**    | **86.6 / 100** |
| **Exception Type**   | Severe Deterioration |
| **Recommended Action** | **Reject/Repurchase** |
| **Top Drivers**      | `CurrentLtv;DaysPastDue;InterestRate` |
| **Days Past Due**    | 23 |
| **Current Balance**  | 52576.79441285193 |

**Audit Narrative:** Loan DOSSIER_012 (TX) identified as a multivariate ML outlier (score: 86.6/100, action: 'Reject/Repurchase'). The unsupervised ensemble detected abnormal patterns in: CurrentLtv, DaysPastDue, InterestRate. Days past due: 23. This loan's risk profile deviates significantly from the training population distribution — manual underwriting review recommended.

---

## Case 08 🔴 — Loan `DOSSIER_015`

| Field                | Value |
|----------------------|-------|
| **Loan ID**          | `DOSSIER_015` |
| **Reporting Month**  | 2023-06-01 |
| **State**            | TX |
| **Vintage**          | 2020 |
| **Servicer**         | Beta Capital |
| **Anomaly Score**    | **86.1 / 100** |
| **Exception Type**   | Severe Deterioration |
| **Recommended Action** | **Reject/Repurchase** |
| **Top Drivers**      | `CurrentLtv;DaysPastDue;InterestRate` |
| **Days Past Due**    | 67 |
| **Current Balance**  | 317749.60009560897 |

**Audit Narrative:** Loan DOSSIER_015 (TX) identified as a multivariate ML outlier (score: 86.1/100, action: 'Reject/Repurchase'). The unsupervised ensemble detected abnormal patterns in: CurrentLtv, DaysPastDue, InterestRate. Days past due: 67. This loan's risk profile deviates significantly from the training population distribution — manual underwriting review recommended.

---

## Case 09 🔴 — Loan `DOSSIER_016`

| Field                | Value |
|----------------------|-------|
| **Loan ID**          | `DOSSIER_016` |
| **Reporting Month**  | 2023-06-01 |
| **State**            | TX |
| **Vintage**          | 2020 |
| **Servicer**         | Alpha Servicing |
| **Anomaly Score**    | **84.8 / 100** |
| **Exception Type**   | Severe Deterioration |
| **Recommended Action** | **Manual Triage** |
| **Top Drivers**      | `CurrentLtv;DaysPastDue;InterestRate` |
| **Days Past Due**    | 67 |
| **Current Balance**  | 156382.53432191425 |

**Audit Narrative:** Loan DOSSIER_016 (TX) identified as a multivariate ML outlier (score: 84.8/100, action: 'Manual Triage'). The unsupervised ensemble detected abnormal patterns in: CurrentLtv, DaysPastDue, InterestRate. Days past due: 67. This loan's risk profile deviates significantly from the training population distribution — manual underwriting review recommended.

---

## Case 10 🟠 — Loan `DOSSIER_006`

| Field                | Value |
|----------------------|-------|
| **Loan ID**          | `DOSSIER_006` |
| **Reporting Month**  | 2023-06-01 |
| **State**            | FL |
| **Vintage**          | 2018 |
| **Servicer**         | Alpha Servicing |
| **Anomaly Score**    | **78.7 / 100** |
| **Exception Type**   | Servicer Discrepancy |
| **Recommended Action** | **Manual Triage** |
| **Top Drivers**      | `CurrentLtv;DaysPastDue;InterestRate` |
| **Days Past Due**    | 56 |
| **Current Balance**  | 65331.31802553007 |

**Audit Narrative:** Loan DOSSIER_006 (FL) flagged for 'Manual Triage' (score: 78.7/100) due to a data conflict between Alpha Servicing and the master record. Reconciliation note: 'Balance delta exceeds tolerance.'. Top contributing factors: CurrentLtv, DaysPastDue, InterestRate. Servicer feed must be reconciled and a corrected tape submitted within 5 business days.

---

## Case 11 🟠 — Loan `DOSSIER_005`

| Field                | Value |
|----------------------|-------|
| **Loan ID**          | `DOSSIER_005` |
| **Reporting Month**  | 2023-06-01 |
| **State**            | FL |
| **Vintage**          | 2019 |
| **Servicer**         | Alpha Servicing |
| **Anomaly Score**    | **78.2 / 100** |
| **Exception Type**   | Servicer Discrepancy |
| **Recommended Action** | **Manual Triage** |
| **Top Drivers**      | `CurrentLtv;DaysPastDue;InterestRate` |
| **Days Past Due**    | 116 |
| **Current Balance**  | 362592.3924627692 |

**Audit Narrative:** Loan DOSSIER_005 (FL) flagged for 'Manual Triage' (score: 78.2/100) due to a data conflict between Alpha Servicing and the master record. Reconciliation note: 'Balance delta exceeds tolerance.'. Top contributing factors: CurrentLtv, DaysPastDue, InterestRate. Servicer feed must be reconciled and a corrected tape submitted within 5 business days.

---

## Case 12 🔴 — Loan `DOSSIER_014`

| Field                | Value |
|----------------------|-------|
| **Loan ID**          | `DOSSIER_014` |
| **Reporting Month**  | 2023-06-01 |
| **State**            | TX |
| **Vintage**          | 2020 |
| **Servicer**         | Alpha Servicing |
| **Anomaly Score**    | **77.9 / 100** |
| **Exception Type**   | Severe Deterioration |
| **Recommended Action** | **Manual Triage** |
| **Top Drivers**      | `CurrentLtv;DaysPastDue;InterestRate` |
| **Days Past Due**    | 16 |
| **Current Balance**  | 90085.52573759071 |

**Audit Narrative:** Loan DOSSIER_014 (TX) identified as a multivariate ML outlier (score: 77.9/100, action: 'Manual Triage'). The unsupervised ensemble detected abnormal patterns in: CurrentLtv, DaysPastDue, InterestRate. Days past due: 16. This loan's risk profile deviates significantly from the training population distribution — manual underwriting review recommended.

---

## Case 13 🔴 — Loan `DOSSIER_010`

| Field                | Value |
|----------------------|-------|
| **Loan ID**          | `DOSSIER_010` |
| **Reporting Month**  | 2023-06-01 |
| **State**            | CA |
| **Vintage**          | 2022 |
| **Servicer**         | Beta Capital |
| **Anomaly Score**    | **77.0 / 100** |
| **Exception Type**   | Severe Deterioration |
| **Recommended Action** | **Manual Triage** |
| **Top Drivers**      | `CurrentLtv;DaysPastDue;InterestRate` |
| **Days Past Due**    | 84 |
| **Current Balance**  | 159328.32448371436 |

**Audit Narrative:** Loan DOSSIER_010 (CA) identified as a multivariate ML outlier (score: 77.0/100, action: 'Manual Triage'). The unsupervised ensemble detected abnormal patterns in: CurrentLtv, DaysPastDue, InterestRate. Days past due: 84. This loan's risk profile deviates significantly from the training population distribution — manual underwriting review recommended.

---

## Case 14 🔴 — Loan `DOSSIER_002`

| Field                | Value |
|----------------------|-------|
| **Loan ID**          | `DOSSIER_002` |
| **Reporting Month**  | 2023-06-01 |
| **State**            | FL |
| **Vintage**          | 2020 |
| **Servicer**         | Beta Capital |
| **Anomaly Score**    | **73.8 / 100** |
| **Exception Type**   | Data Logic Error |
| **Recommended Action** | **Reject/Repurchase** |
| **Top Drivers**      | `CurrentLtv;DaysPastDue;InterestRate` |
| **Days Past Due**    | 44 |
| **Current Balance**  | 374367.7460970106 |

**Audit Narrative:** Loan DOSSIER_002 (FL) flagged for 'Reject/Repurchase' with anomaly score 73.8/100 due to deterministic accounting violations: CurrentLtv, DaysPastDue, InterestRate. Current balance reported as $374,367.75 against an original balance of $400,000.00, violating balance integrity constraints. Immediate data correction and servicer confirmation required before pool reporting.

---

## Case 15 🔴 — Loan `DOSSIER_001`

| Field                | Value |
|----------------------|-------|
| **Loan ID**          | `DOSSIER_001` |
| **Reporting Month**  | 2023-06-01 |
| **State**            | TX |
| **Vintage**          | 2022 |
| **Servicer**         | Beta Capital |
| **Anomaly Score**    | **72.8 / 100** |
| **Exception Type**   | Data Logic Error |
| **Recommended Action** | **Reject/Repurchase** |
| **Top Drivers**      | `CurrentLtv;DaysPastDue;InterestRate` |
| **Days Past Due**    | 91 |
| **Current Balance**  | 325122.50684693386 |

**Audit Narrative:** Loan DOSSIER_001 (TX) flagged for 'Reject/Repurchase' with anomaly score 72.8/100 due to deterministic accounting violations: CurrentLtv, DaysPastDue, InterestRate. Current balance reported as $325,122.51 against an original balance of $400,000.00, violating balance integrity constraints. Immediate data correction and servicer confirmation required before pool reporting.

---

## Case 16 🔴 — Loan `DOSSIER_019`

| Field                | Value |
|----------------------|-------|
| **Loan ID**          | `DOSSIER_019` |
| **Reporting Month**  | 2023-06-01 |
| **State**            | TX |
| **Vintage**          | 2021 |
| **Servicer**         | Alpha Servicing |
| **Anomaly Score**    | **71.5 / 100** |
| **Exception Type**   | Severe Deterioration |
| **Recommended Action** | **Manual Triage** |
| **Top Drivers**      | `CurrentLtv;DaysPastDue;InterestRate` |
| **Days Past Due**    | 66 |
| **Current Balance**  | 324364.37318724475 |

**Audit Narrative:** Loan DOSSIER_019 (TX) identified as a multivariate ML outlier (score: 71.5/100, action: 'Manual Triage'). The unsupervised ensemble detected abnormal patterns in: CurrentLtv, DaysPastDue, InterestRate. Days past due: 66. This loan's risk profile deviates significantly from the training population distribution — manual underwriting review recommended.

---

## Case 17 🟠 — Loan `DOSSIER_008`

| Field                | Value |
|----------------------|-------|
| **Loan ID**          | `DOSSIER_008` |
| **Reporting Month**  | 2023-06-01 |
| **State**            | FL |
| **Vintage**          | 2019 |
| **Servicer**         | Alpha Servicing |
| **Anomaly Score**    | **67.8 / 100** |
| **Exception Type**   | Servicer Discrepancy |
| **Recommended Action** | **Manual Triage** |
| **Top Drivers**      | `CurrentLtv;DaysPastDue;InterestRate` |
| **Days Past Due**    | 56 |
| **Current Balance**  | 116314.9756795 |

**Audit Narrative:** Loan DOSSIER_008 (FL) flagged for 'Manual Triage' (score: 67.8/100) due to a data conflict between Alpha Servicing and the master record. Reconciliation note: 'Balance delta exceeds tolerance.'. Top contributing factors: CurrentLtv, DaysPastDue, InterestRate. Servicer feed must be reconciled and a corrected tape submitted within 5 business days.

---

## Case 18 🟠 — Loan `DOSSIER_007`

| Field                | Value |
|----------------------|-------|
| **Loan ID**          | `DOSSIER_007` |
| **Reporting Month**  | 2023-06-01 |
| **State**            | FL |
| **Vintage**          | 2021 |
| **Servicer**         | Beta Capital |
| **Anomaly Score**    | **63.7 / 100** |
| **Exception Type**   | Servicer Discrepancy |
| **Recommended Action** | **Manual Triage** |
| **Top Drivers**      | `CurrentLtv;DaysPastDue;InterestRate` |
| **Days Past Due**    | 89 |
| **Current Balance**  | 388628.4063519735 |

**Audit Narrative:** Loan DOSSIER_007 (FL) flagged for 'Manual Triage' (score: 63.7/100) due to a data conflict between Beta Capital and the master record. Reconciliation note: 'Balance delta exceeds tolerance.'. Top contributing factors: CurrentLtv, DaysPastDue, InterestRate. Servicer feed must be reconciled and a corrected tape submitted within 5 business days.

---

## Case 19 🟠 — Loan `DOSSIER_009`

| Field                | Value |
|----------------------|-------|
| **Loan ID**          | `DOSSIER_009` |
| **Reporting Month**  | 2023-06-01 |
| **State**            | FL |
| **Vintage**          | 2020 |
| **Servicer**         | Alpha Servicing |
| **Anomaly Score**    | **63.1 / 100** |
| **Exception Type**   | Servicer Discrepancy |
| **Recommended Action** | **Manual Triage** |
| **Top Drivers**      | `CurrentLtv;DaysPastDue;InterestRate` |
| **Days Past Due**    | 27 |
| **Current Balance**  | 284434.89813887863 |

**Audit Narrative:** Loan DOSSIER_009 (FL) flagged for 'Manual Triage' (score: 63.1/100) due to a data conflict between Alpha Servicing and the master record. Reconciliation note: 'Balance delta exceeds tolerance.'. Top contributing factors: CurrentLtv, DaysPastDue, InterestRate. Servicer feed must be reconciled and a corrected tape submitted within 5 business days.

---

## Case 20 🔴 — Loan `DOSSIER_017`

| Field                | Value |
|----------------------|-------|
| **Loan ID**          | `DOSSIER_017` |
| **Reporting Month**  | 2023-06-01 |
| **State**            | TX |
| **Vintage**          | 2020 |
| **Servicer**         | Beta Capital |
| **Anomaly Score**    | **61.2 / 100** |
| **Exception Type**   | Severe Deterioration |
| **Recommended Action** | **Manual Triage** |
| **Top Drivers**      | `CurrentLtv;DaysPastDue;InterestRate` |
| **Days Past Due**    | 25 |
| **Current Balance**  | 192985.02530362265 |

**Audit Narrative:** Loan DOSSIER_017 (TX) identified as a multivariate ML outlier (score: 61.2/100, action: 'Manual Triage'). The unsupervised ensemble detected abnormal patterns in: CurrentLtv, DaysPastDue, InterestRate. Days past due: 25. This loan's risk profile deviates significantly from the training population distribution — manual underwriting review recommended.

---
