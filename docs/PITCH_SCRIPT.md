# Intain-Sight: Video Pitch Scripts & Presentation Guide

This guide contains the pitch scripts used for hackathon video demonstrations, ranging from ultra-condensed to fully technical, plus YouTube upload metadata.

---

## 🎬 Script 1: The Relaxed Pace Script (~3 Minutes / ~300 Words)
*Designed for a calm, conversational presentation where the UI does the talking.*

**[Start on Page 1 - Pool Overview]**
"Hi judges, I'm Khush, and this is **Intain-Sight**. We wanted to move past black-box machine learning that just guesses 'yes or no' on loan defaults. 

Here on the Pool Overview, you'll see our first novelty: we upgraded to a **Competing Risks Framework**. Using LightGBM, we simultaneously predict both default *and* prepayment, giving a much truer quantitative view of the portfolio."

**[Click to Page 2 - Loan Explorer]**
"If we explore a specific loan, SHAP values explain exactly what is driving its risk. 

But our real breakthrough here is **Prescriptive AI**. If a loan is too risky, the system doesn't just reject it—it calculates the exact mathematical offset to fix it, like telling the underwriter to lower the balance by ten thousand dollars."

**[Click to Page 3 - Scenario Sandbox]**
"In our **Scenario Sandbox**, portfolio managers can stress-test the data using a Monte Carlo simulator. You simply adjust macroeconomic shocks—like Interest Rates—and watch exactly how different loan vintages react over the next twelve months."

**[Click to Page 4 - Verification Queue]**
"For strange edge cases, we deployed an **Isolation Forest** scanning a 4D feature space. It catches hidden statistical anomalies that normal business rules miss, routing them here to the Verification Queue for manual review."

**[Click to Page 5 - LLM Copilot]**
"To assist that review, underwriters can chat directly with our **Gemini-powered Copilot**. And because finance is strictly regulated, every AI interaction is permanently logged in an immutable Compliance Ledger, backed by mathematical guardrails to block hallucinations."

**[Click to Page 6 - Drift Error Analysis]**
"Finally, our **Drift Analysis** uses Population Stability Index to ensure our data isn't degrading. Notice the AI text boxes? The dashboard mathematically parses the charts on the screen and streams the live vectors to Gemini for real-time risk assessments. 

Getting this live was a challenge. We built a strict CI/CD pipeline, heavily optimized our packages to beat cloud memory limits, and engineered a secure secrets vault after Google's security bots quarantined our project during deployment. It was a battlefield, but the platform is live and secure. Thank you!"

---

## ⏱️ Script 2: The Balanced "Goldilocks" Script (~4 Minutes / ~450 Words)
*Balances technical rigor (naming exact algorithms) with direct business value.*

**[Start on Page 1 - Pool Overview]**
"Hi judges, I'm Khush, and this is **Intain-Sight**. In securitization, underwriters usually rely on black-box machine learning that just spits out a simple Yes or No for loan defaults. We wanted to build an enterprise-grade platform that is fully transparent and actionable. 

Here on the **Pool Overview**, you can see our first major novelty. Instead of basic binary classification, we engineered a **Competing Risks Framework** using an ensemble of LightGBM classifiers. It simultaneously predicts if a borrower will default *or* prepay their mortgage, which gives a much more accurate quantitative picture of the portfolio."

**[Click to Page 2 - Loan Explorer]**
"If we click into the **Loan Explorer**, we don't just hand the underwriter a black-box risk score. We use SHAP values to generate this waterfall chart, showing exactly what data points are driving the risk. 

But our biggest feature here is **Actionable Counterfactuals**—what we call Prescriptive AI. If a loan is too risky, the system doesn't just reject it. It dynamically calculates the math and prescribes a fix. It literally tells the underwriter: *'Lower the balance by this exact amount to make the loan safe.'*"

**[Click to Page 3 - Scenario Sandbox]**
"Next is the **Scenario Sandbox**. We built a vector-optimized Monte Carlo simulator. This allows portfolio managers to stress-test the data. You can drag these sliders for macroeconomic shocks—like Unemployment or Interest Rates—and see exactly how different vintages of loans will hold up over the next 12 months."

**[Click to Page 4 - Verification Queue]**
"But what about weird loans that slip past normal underwriting rules? We deployed an unsupervised machine learning model—an **Isolation Forest**—scanning across a 4-dimensional feature space. It catches hidden statistical anomalies and pushes them right here to the **Verification Queue** so a human can manually triage them."

**[Click to Page 5 - LLM Copilot]**
"To assist with that manual review, we integrated a live **LLM Copilot** using Google's Gemini-3.6-Flash API. Underwriters can chat directly with the AI to investigate loan files. Because finance is heavily regulated, every single AI decision is hashed and tracked in this immutable Compliance Ledger, and we wrote strict mathematical guardrails to intercept any AI hallucinations before the user sees them."

**[Click to Page 6 - Drift Error Analysis]**
"Finally, our **Drift Analysis** page uses the Population Stability Index to ensure our data isn't degrading over time. Notice these text boxes at the bottom? The dashboard actually parses the live mathematical vectors on the screen and streams them to Gemini, generating real-time, written risk assessments. 

Getting this live tested our DevOps skills. We built a full CI/CD pipeline with GitHub Actions, heavily optimized our Python wheels to bypass Streamlit's strict cloud memory limits, and even had to build a secure vault system after Google's security bots quarantined our cloud project during a live deployment. But we survived the deployment battlefield, the app is live, and we're incredibly proud of it. Thank you!"

---

## 📺 YouTube Metadata

### Title
> Intain-Sight: Enterprise AI Verification Agent | Hackathon Demo

### Description
Intain-Sight is an enterprise-grade AI Verification Agent built to bridge the gap between black-box machine learning and explainable securitization. Designed to move beyond basic binary classification, this platform provides actionable, quantitative intelligence for mortgage risk and underwriting.

- **GitHub Repository:** https://github.com/Quirrells-Quill/Intain_Challenge_AI
- **Live Demo:** https://intainchallengeai-rktvwbztxdbndxbvbkguwc.streamlit.app/

**Key Architectural Novelties:**
- **Competing Risks Framework:** Simultaneous 12-month Default and Prepayment predictions using LightGBM ensembles.
- **Prescriptive AI (Actionable Counterfactuals):** Uses SHAP values to calculate the exact mathematical offset needed to fix high-risk loans.
- **Unsupervised Anomaly Triage:** A 4D Isolation Forest that catches hidden statistical edge cases that evade standard boolean business rules.
- **Live GenAI Copilot:** Integrates Google's Gemini-3.6-Flash API alongside an immutable Compliance Ledger to parse live mathematical vectors and assist underwriters securely.
- **Enterprise DevOps:** Fully containerized with a strict GitHub Actions CI/CD pipeline, time-series out-of-time validation, and secure secrets vaulting.

**Video Timestamps:**
- 0:00 - Introduction & Competing Risks Framework
- 0:35 - Loan Explorer & Prescriptive AI (SHAP)
- 1:10 - Scenario Sandbox & Macro Shocks
- 1:40 - Verification Queue (Isolation Forest)
- 2:10 - LLM Copilot & Compliance Ledger
- 2:40 - Drift Analysis & CI/CD Deployment Architecture
