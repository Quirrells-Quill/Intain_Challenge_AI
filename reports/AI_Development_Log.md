# INTAIN-SIGHT: Agentic AI Development Log
> **Mandatory Governance Deliverable**: Programmatic verification of AI vs Human development attribution.

## Section 1: AI Tools Utilized
- **Primary AI Agent**: Antigravity (Powered by Gemini Pro / Deepmind Architecture)
- **Tasks Handled**: Boilerplate generation, Polars vectorization, Multi-task ML architecture, Pytest formulation, and Plotly visualization.

## Section 2: Agentic Code Share Estimate
Calculated via the `src/governance/ai_tracker.py` heuristic over the `src/` directory:
- **Total Lines of Code Scanned**: 7287
- **AI-Generated / Auto-Scaffolded (Est)**: 85.0% (6193 lines)
- **Human-Refactored / Authored (Est)**: 15.0% (1093 lines)

## Section 3: Human Review & Governance Process
All Agentic outputs were evaluated via **Strict Test-Driven Validation**. Code was evaluated against:
1. **Temporal Leakage Rules**: Enforcing time-aware chronological grouping.
2. **API Completeness**: Ensuring correct version matching (e.g. `lifelines` AalenJohansenFitter APIs).
3. **Memory/Performance**: Vectorized Polars broadcasting to avoid standard python iterative loops.

## Section 4: Representative Prompt Log
The following chronological log details the interplay between the human architect and the AI Agent.

### Prompt 1: Stage 1: Data Intelligence
**Request**: *"Implement the Data Intelligence module in `src/profiling/`. Calculate securitization metrics (WAC, WAM) and detect data drift using Polars."*
**AI Tool**: Antigravity (Gemini Pro)
**Status**: `ACCEPTED`
**Human-in-the-Loop Review**: Clean Polars aggregations. Handled weighted averages correctly.

### Prompt 2: Stage 2: Feature Engineering
**Request**: *"Use `polars` window functions grouping by `loan_id` and ordering by `reporting_month` to create temporal rolling window features like 3-month rolling DPD."*
**AI Tool**: Antigravity (Gemini Pro)
**Status**: `MODIFIED`
**Human-in-the-Loop Review**: AI generated standard rolling windows but missed sorting by date prior to grouping. Human added `.sort('reporting_month')` before `rolling()`.

### Prompt 3: Stage 3: Predictive Modeling
**Request**: *"Implement an Ensemble GBDT Architecture (XGBoost, LightGBM) to predict 12-month default. Handle imbalanced classes using scale_pos_weight."*
**AI Tool**: Antigravity (Gemini Pro)
**Status**: `REJECTED`
**Error Mode Detected**: Failed temporal leakage constraint
**Human-in-the-Loop Review**: AI used standard train_test_split. Human rejected and rewrote using a strict time-aware grouped split to prevent temporal data leakage.

### Prompt 4: Stage 4: Survival Modeling
**Request**: *"Fit a Fine-Gray competing risk model evaluating Default vs. Prepayment using lifelines AalenJohansenFitter."*
**AI Tool**: Antigravity (Gemini Pro)
**Status**: `REJECTED`
**Error Mode Detected**: Deprecated API argument usage
**Human-in-the-Loop Review**: AI passed `cause_of_interest` to `__init__`, which is deprecated in newer versions of lifelines. Human corrected the API to pass `event_of_interest` into the `fit()` method.

### Prompt 5: Stage 5: Anomaly Detection
**Request**: *"Ingest configs/validation_rules.json to flag hard deterministic accounting violations like negative balances."*
**AI Tool**: Antigravity (Gemini Pro)
**Status**: `ACCEPTED`
**Human-in-the-Loop Review**: Correctly parsed the JSON rules and vectorized the bounds checking logic.

### Prompt 6: Stage 6: Monte Carlo Scenarios
**Request**: *"Implement a 1,000-iteration Monte Carlo simulation engine with correlated parameter perturbation for macro scenarios."*
**AI Tool**: Antigravity (Gemini Pro)
**Status**: `REJECTED`
**Error Mode Detected**: High memory consumption loop (Performance)
**Human-in-the-Loop Review**: AI attempted to loop row-by-row in Python leading to OOM and extreme latency. Human rewrote the core to use NumPy broadcasting over the entire pool simultaneously.

### Prompt 7: Stage 7: Explainability
**Request**: *"Compute Global TreeSHAP across the multi-task prediction targets and generate Plotly bar charts."*
**AI Tool**: Antigravity (Gemini Pro)
**Status**: `MODIFIED`
**Human-in-the-Loop Review**: AI returned 3D arrays for random forest SHAP values. Human added explicit array slicing `shap_vals[:, :, 1]` to extract the correct target class.

### Prompt 8: Stage 7: Counterfactuals
**Request**: *"Build a Counterfactual Explanation Engine determining the minimum actionable feature perturbation required to transition a loan to approved."*
**AI Tool**: Antigravity (Gemini Pro)
**Status**: `ACCEPTED`
**Human-in-the-Loop Review**: Excellent greedy search implementation respecting immutable features like vintage and loan_id.

### Prompt 9: Stage 8: LLM Copilot
**Request**: *"Create a Gemini API wrapper that enforces structured JSON output using Pydantic schemas."*
**AI Tool**: Antigravity (Gemini Pro)
**Status**: `MODIFIED`
**Human-in-the-Loop Review**: AI successfully used generation_config. Human injected a mock_mode fallback so CI/CD smoke tests pass without an active internet connection.

### Prompt 10: Stage 8: Guardrails
**Request**: *"Write a guardrail to regex-scan LLM narrative text for numerical claims and verify they match the actual ML inference probability within a 0.03 tolerance."*
**AI Tool**: Antigravity (Gemini Pro)
**Status**: `ACCEPTED`
**Human-in-the-Loop Review**: Solid regex extraction handling both explicit decimal floats and percentage string formats seamlessly.

## Section 5: MLflow Experiment Link
**Tracking URI**: `http://localhost:5000` *(Placeholder: Reference MLflow server UI for artifact tracking)*

## Section 6: Lessons Learned
- **What Worked**: Auto-generating Polars schema aggregations, boilerplate class architectures, and Plotly UI components provided 10x velocity.
- **What Failed**: The AI struggled occasionally with edge-case financial mathematical definitions (like passing `cause_of_interest` in older versions of `lifelines`) and required prompt correction on memory-heavy `for-loops`.
- **Correction Velocity**: Providing exact traceback errors to the AI yielded near-instant deterministic fixes, proving that Agentic workflows shine when paired with rigid QA smoke tests.
## Phase 2: Enterprise CI/CD & Cloud Deployment (Final Stage)

### Prompt 11: CI/CD Pipeline Generation
**Request**: *"Write a GitHub Actions pipeline that installs dependencies and runs the Pytest compliance harness to prevent PR regression."*
**AI Tool**: Antigravity (Gemini Pro)
**Status**: ACCEPTED
**Human-in-the-Loop Review**: Human had to modify the PYTHONPATH context for Pytest in the Ubuntu runner to ensure src module resolution.

### Prompt 12: Streamlit Cloud Dependency Resolution
**Request**: *"Streamlit cloud is failing to compile Pillow from source due to zlib headers. Fix the requirements."*
**AI Tool**: Antigravity (Gemini Pro)
**Status**: ACCEPTED
**Human-in-the-Loop Review**: AI correctly identified the root cause as strict version pinning against incompatible Python 3.12 cloud environments. Unpinned requirements enabled pre-compiled wheels, reducing cloud boot time from failure to <60 seconds.

### Prompt 13: Secrets Management & Security
**Request**: *"The Gemini API key threw a 403 error because it was leaked on GitHub. Secure the app."*
**AI Tool**: Antigravity (Gemini Pro)
**Status**: ACCEPTED
**Human-in-the-Loop Review**: AI stripped hardcoded keys from components.py and 5_LLM_Copilot.py, implementing st.secrets vault integration to satisfy enterprise security protocols.
