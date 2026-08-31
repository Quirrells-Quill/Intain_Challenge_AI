it# 🏦 Intain-Sight: Verification Agent
> **Enterprise Loan Performance Intelligence Engine** | Intain FinTech Challenge 2026

Intain-Sight is an end-to-end AI underwriting pipeline that seamlessly integrates deterministic accounting rules, multi-task machine learning, survival analysis, stochastic Monte Carlo stress testing, and a highly governed LLM Copilot.

## 🚀 Quickstart

1. **Run the Full Pipeline**:
   Compiles predictions, runs validations, and generates the final strict `submission.csv`.
   ```bash
   python src/pipeline/run_all.py --fast-dev
   ```

2. **Launch the Dashboard**:
   Interact with the Streamlit Reviewer Workspace.
   ```bash
   streamlit run src/dashboard/app.py
   ```

## 🏗️ Architecture Pillars (The 5 Core Mandates)

1. **Schema Immutability**: Flawless compilation of `submission.csv` via the `FinalSubmissionCompiler`. Zero nulls, strict types.
2. **Deterministic & ML Fusion**: Merging JSON-defined hard validation rules with Isolation Forests and GBDT anomaly detection.
3. **Temporal Leakage Prevention**: Strict 12-month grouped chronological train/test splits.
4. **Governed AI Copilot**: RAG-augmented Gemini API restricted by an independent numerical guardrail that catches and flags hallucinations.
5. **Agentic Code Log**: Programmatically tracked via `reports/AI_Development_Log.md` demonstrating human-in-the-loop ML ops.

## 📁 Repository Structure

- `configs/`: Centralized JSON dictionaries, macro-scenario matrices, and validation rules.
- `src/pipeline/`: E2E Orchestration and `submission.csv` compiler.
- `src/dashboard/`: Streamlit React UI.
- `src/models/`: Multi-Task GBDT ensembles.
- `src/survival/`: Aalen-Johansen CIF and Cox Proportional Hazards.
- `src/scenarios/`: Vectorized Monte Carlo simulation engines.
- `src/explain/`: Local/Global TreeSHAP and Actionable Counterfactual engines.
- `src/llm/`: Pydantic-enforced LLM wrapper, TF-IDF RAG, numerical guardrails, and SQLite auditing.
- `reports/`: Governance, AI tracking, and Failure Analysis deliverables.

## 📄 Submission Compliance

This project has been rigorously tested against the provided rubric and passes all Smoke Tests covering temporal CV constraints, API logic, schema checks, and hallucination bounds.
