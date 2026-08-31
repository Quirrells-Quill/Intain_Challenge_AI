"""
Explainability Pipeline Orchestrator — explain_pipeline.py

Ties together global SHAP, local SHAP, counterfactual generation, uncertainty 
quantification, and fairness auditing into a single end-to-end executable flow.
"""

import os
import polars as pl
from typing import Dict, Any, List, Callable
from src.utils.logger import get_logger

from src.explain.global_explainer import GlobalModelExplainer
from src.explain.local_explainer import LocalLoanExplainer
from src.explain.counterfactuals import CounterfactualEngine
from src.explain.uncertainty import EnsembleUncertaintyEstimator
from src.explain.fairness_auditor import SubgroupFairnessAuditor

logger = get_logger(__name__)


class ExplainabilityPipeline:
    def __init__(
        self, 
        models: Dict[str, Any], 
        ensemble_predictors: List[Callable[[pl.DataFrame], pl.DataFrame]],
        features: List[str]
    ):
        self.models = models
        self.ensemble_predictors = ensemble_predictors
        self.features = features
        
        self.reports_dir = "reports"
        self.figures_dir = os.path.join(self.reports_dir, "figures", "explain")
        self.data_dir = os.path.join("data", "processed")
        
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    def run(self, df: pl.DataFrame, target_col: str = "default_flag"):
        """
        Executes the explainability pipeline.
        """
        logger.info("==================================================")
        logger.info("STARTING EXPLAINABILITY PIPELINE")
        logger.info("==================================================")
        
        # 1. Global Explainer
        bg_data = df.sample(n=min(500, df.height), seed=42)
        global_explainer = GlobalModelExplainer(self.models, bg_data)
        global_explainer.compute_shap()
        global_explainer.export_summary_plots(self.figures_dir)
        
        # 2. Local Explainer & Contrastive Attributions
        local_explainer = LocalLoanExplainer(global_explainer.explainers, self.features)
        
        # Pick one high-risk loan for a local waterfall example
        if "prob_default" not in df.columns:
            # We assume df has prob_default, if not, generate it from ensemble[0]
            df = self.ensemble_predictors[0](df)
            
        high_risk_loans = df.filter(pl.col("prob_default") >= 0.35)
        local_examples = []
        contrastive_examples = []
        
        if high_risk_loans.height > 0:
            sample_loan = high_risk_loans.head(1)
            explanation = local_explainer.explain_loan(sample_loan, "default")
            local_explainer.export_local_waterfall(
                explanation, 
                os.path.join(self.figures_dir, "local_waterfall_example.html")
            )
            local_examples.append((sample_loan["loan_id"][0], explanation))
            
            # Contrastive
            if "prepay" in self.models:
                contrastive = local_explainer.explain_contrastive(sample_loan, "default", "prepay")
                contrastive_examples.append((sample_loan["loan_id"][0], contrastive))

        # 3. Counterfactual Engine
        cf_engine = CounterfactualEngine(self.ensemble_predictors[0], target_col="prob_default", threshold=0.10)
        cf_prescriptions = []
        for row in high_risk_loans.head(5).iter_rows(named=True):
            single_df = pl.DataFrame([row])
            res = cf_engine.generate_counterfactual_prescription(single_df)
            res["loan_id"] = row["loan_id"]
            cf_prescriptions.append(res)
            
        # 4. Epistemic Uncertainty Estimation
        uncertainty_engine = EnsembleUncertaintyEstimator(self.ensemble_predictors, target_col="prob_default")
        df_uncertainty = uncertainty_engine.estimate_uncertainty(df)
        uncertainty_engine.generate_uncertainty_heatmap(
            df_uncertainty, 
            os.path.join(self.figures_dir, "uncertainty_heatmap.html")
        )
        
        # Save features for Submission Builder
        out_parquet = os.path.join(self.data_dir, "test_explainability_features.parquet")
        df_uncertainty.write_parquet(out_parquet)
        logger.info(f"Persisted explainability features to {out_parquet}")

        # 5. Fairness Audit
        if target_col in df_uncertainty.columns:
            auditor = SubgroupFairnessAuditor(target_col=target_col, prob_col="ensemble_mean")
            fairness_ledger = auditor.audit(df_uncertainty)
        else:
            fairness_ledger = {}
            logger.warning(f"Target col '{target_col}' missing. Skipping fairness audit.")

        # 6. Generate Report
        self._generate_report(cf_prescriptions, fairness_ledger)
        
        logger.info("==================================================")
        logger.info("EXPLAINABILITY PIPELINE COMPLETE")
        logger.info("==================================================")

    def _generate_report(self, cf_prescriptions: List[Dict], fairness_ledger: Dict[str, pl.DataFrame]):
        md_lines = [
            "# INTAIN-SIGHT: Explainability & Counterfactual Report",
            "",
            "> **CONFIDENTIAL**: Auto-generated by Verification Agent Explainability Subsystem",
            "",
            "## 1. Actionable Counterfactual Prescriptions",
            "Minimal-distance interventions to rescue high-risk loans (Predicted Default -> Target < 10%).",
            ""
        ]
        
        if not cf_prescriptions:
            md_lines.append("_No high-risk loans requiring counterfactuals found._")
        else:
            for cf in cf_prescriptions:
                md_lines.append(f"### Loan ID: {cf['loan_id']}")
                md_lines.append(f"- **Initial Default Prob**: {cf['orig_prob']*100:.1f}%")
                if cf['success']:
                    md_lines.append(f"- **Target Default Prob Achieved**: {cf['final_prob']*100:.1f}%")
                    md_lines.append("- **Recommended Actions**:")
                    for action in cf['prescription']:
                        md_lines.append(f"  - {action}")
                else:
                    md_lines.append("- **Status**: Rescue failed within business constraints.")
                md_lines.append("")

        md_lines.extend([
            "## 2. Fairness & Subgroup Disparity Audit",
            "Highlights demographic or geographic segments with False Positive Rates (FPR) exceeding 1.5x the portfolio baseline.",
            ""
        ])
        
        flags_found = False
        for group, df_audit in fairness_ledger.items():
            flagged = df_audit.filter(pl.col("flagged_disparity") == True)
            if flagged.height > 0:
                flags_found = True
                md_lines.append(f"### {group.replace('_', ' ').title()}")
                for row in flagged.iter_rows(named=True):
                    md_lines.append(
                        f"- **{row['value']}**: FPR = {row['fpr']*100:.1f}% "
                        f"(Disparate Impact Ratio = {row['dir']:.2f})"
                    )
                md_lines.append("")
                
        if not flags_found:
            md_lines.append("_All subgroup FPRs remain within acceptable tolerance bounds._")
            
        report_path = os.path.join(self.reports_dir, "EXPLAINABILITY_REPORT.md")
        with open(report_path, "w") as f:
            f.write("\n".join(md_lines))
        logger.info(f"Report generated at {report_path}")
