# Intain-Sight: LLM Governance & Guardrail Audit
Demonstration of the GuardrailEngine intercepting structural and numerical hallucinations.

## Base Context Provided to LLM
```json
{'loan_data': {'current_balance': 250000.0, 'interest_rate': 5.5}, 'anomaly_data': {'anomaly_score': 85}, 'shap_data': {'top_drivers': ['interest_rate']}}
```

### Test Case 1: Gross Numerical Hallucination
**LLM Output Summary**: The borrower has a balance of $300,000.
**Guardrail Result**: Passed? False
**Violations**: ["Hallucination detected: Number '300' not found in factual context."]

### Test Case 2: Subtle Decimal Hallucination
**LLM Output Summary**: Interest rate is currently 5.8%.
**Guardrail Result**: Passed? False
**Violations**: ["Hallucination detected: Number '5.8' not found in factual context."]

### Test Case 3: Factually Grounded Output
**LLM Output Summary**: The borrower has a balance of $250000.0 and interest rate of 5.5.
**Guardrail Result**: Passed? True
**Violations**: []