# 🏦 Intain-Sight: Enterprise Verification Agent

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3.0-brightgreen.svg)
![Gemini](https://img.shields.io/badge/Gemini-3.6%20Flash-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Intain-Sight** is an enterprise-grade Quantitative Risk & Verification platform designed for the securitization industry. Built during the Intain Hackathon, this system replaces black-box ML predictions with auditable, explainable, and actionable quantitative intelligence.

## 🌟 The Novelty (Why This Project Stands Out)
While standard risk models output a binary prediction (Default / Not Default), this platform is engineered for real-world high-frequency securitization:
1. **Competing Risks Framework**: Simultaneously models 12-month Default and Prepayment probabilities (competing financial hazards) using ensemble LightGBM classifiers.
2. **Actionable Counterfactuals**: Doesn't just flag high-risk loans; it prescribes exact mathematical offsets (e.g., *"Lower balance by $12,500 to drop risk below 10%"*).
3. **Live AI Quant Analyst (Gemini)**: The UI isn't static. It parses the mathematical vectors of the displayed charts and feeds them live to Google's `gemini-3.6-flash`, streaming back professional hedge-fund-style risk takeaways in real time.
4. **Unsupervised Outlier Triage**: Deploys an `IsolationForest` across a 4-dimensional feature space to flag statistical anomalies that evade standard boolean business rules.

## 📁 Repository Architecture
> **Note:** To maintain a lightweight repository and ensure CI/CD reproducibility, there are no static `.pkl` model weights or `.ipynb` notebooks in this repo. The entire training loop is strictly programmatic. 

* `src/data/generator.py`: A High-Fidelity Synthetic Data Generator that mathematically mimics Fannie Mae covariance matrices.
* `src/pipeline/run_all.py`: The master orchestration script. Generates data, trains LightGBM in memory, extracts SHAP matrices, and exports the strict 13-column `master_pool.parquet`.
* `src/dashboard/app.py`: The Bloomberg-Terminal inspired Streamlit Dashboard.
* `tests/test_master_compliance.py`: Strict Pytest harness verifying temporal leakage prevention, tensor shapes, and LLM hallucination guardrails.

## 🚀 Quickstart (Local Run)

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Execute the E2E ML Pipeline**
*(This generates the data, trains the models, and creates the artifacts in ~3 seconds)*
```bash
export PYTHONPATH="."
python src/pipeline/run_all.py
```

**3. Launch the Dashboard**
```bash
streamlit run src/dashboard/app.py
```

## 🐳 Docker Deployment
```bash
docker build -t intain-sight .
docker run -p 8501:8501 intain-sight
```

## 🔒 Security & Compliance
- **Data Governance**: Strict out-of-time (OOT) validation segmenting 2022 vintages to prove the model hasn't overfit to historical macro-environments.
- **LLM Grounding**: The Gemini Copilot is sandboxed to financial QA and explicitly banner-flagged as a recommendation engine, maintaining human-in-the-loop oversight.
