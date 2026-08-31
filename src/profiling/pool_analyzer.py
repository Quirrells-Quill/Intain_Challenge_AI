"""
Data Intelligence & Pool-Level Profiling Module

This module implements the 'Verification Agent' for the Intain-Sight engine.
It analyzes individual loan records and the aggregate asset pool to calculate
securitization metrics, detect data anomalies, and produce a data quality report.
"""

import polars as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
from plotly.subplots import make_subplots

from src.utils.logger import get_logger

logger = get_logger(__name__)

class PoolIntelligenceAnalyzer:
    """
    Verification Agent logic: Analyzes asset pools, detects data drift, 
    reconciles multi-source data, and generates Data Quality (DQ) scores.
    """

    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        logger.info("Initialized PoolIntelligenceAnalyzer")

    def calculate_pool_metrics(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        1. Aggregate Pool-Level Profiling.
        Calculates total loan count, total principal balance, Weighted Average Coupon (WAC), 
        and Weighted Average Maturity (WAM). 
        Identifies Risk Concentration.
        """
        logger.info("Calculating aggregate pool-level metrics...")
        
        total_loan_count = df.height
        total_principal = df["current_balance"].sum() if "current_balance" in df.columns else 0.0
        
        # Weighted Average Coupon (WAC)
        if total_principal > 0 and "interest_rate" in df.columns:
            wac = (df["interest_rate"] * df["current_balance"]).sum() / total_principal
        else:
            wac = 0.0
            
        # Weighted Average Maturity (WAM)
        if total_principal > 0 and "remaining_months_to_maturity" in df.columns:
            wam = (df["remaining_months_to_maturity"] * df["current_balance"]).sum() / total_principal
        else:
            wam = 0.0

        # Risk Concentration (Top State, Credit Band, Vintage)
        risk_concentration = {}
        for col in ["state", "credit_band", "origination_vintage"]:
            if col in df.columns and "current_balance" in df.columns:
                counts = df.group_by(col).agg(
                    pl.col("current_balance").sum().alias("total_balance")
                ).with_columns(
                    (pl.col("total_balance") / total_principal * 100).alias("concentration_pct")
                ).sort("concentration_pct", descending=True)
                
                top_concentration = counts.head(1).to_dicts()
                risk_concentration[col] = top_concentration[0] if top_concentration else None

        self.metrics["total_loan_count"] = total_loan_count
        self.metrics["total_principal_balance"] = total_principal
        self.metrics["wac"] = wac
        self.metrics["wam"] = wam
        self.metrics["risk_concentration"] = risk_concentration
        
        logger.info(f"Pool Metrics Calculated: WAC={wac:.2f}%, WAM={wam:.2f} months")
        return self.metrics

    def verify_servicer_updates(
        self, 
        master_df: pl.DataFrame, 
        servicer_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        2. Multi-Source Reconciliation.
        Cross-checks master performance dataset against servicer updates using 'loan_id'.
        Flags conflicting data points.
        """
        logger.info("Reconciling master performance data with servicer updates...")
        
        if "loan_id" not in master_df.columns or "loan_id" not in servicer_df.columns:
            logger.warning("Missing 'loan_id' for reconciliation.")
            return master_df.with_columns(pl.lit(False).alias("servicer_conflict_flag"))

        joined = master_df.join(
            servicer_df, 
            on="loan_id", 
            how="left", 
            suffix="_servicer"
        )
        
        conflict_conds = []
        
        if "current_balance_servicer" in joined.columns and "current_balance" in joined.columns:
            # We allow a small tolerance (1.0) for floating point balance issues
            conflict_balance = (
                joined["current_balance_servicer"].is_not_null() & 
                ((joined["current_balance"] - joined["current_balance_servicer"]).abs() > 1.0)
            )
            conflict_conds.append(conflict_balance)
            
        if "delinquency_status_servicer" in joined.columns and "delinquency_status" in joined.columns:
            conflict_delinq = (
                joined["delinquency_status_servicer"].is_not_null() & 
                (joined["delinquency_status"] != joined["delinquency_status_servicer"])
            )
            conflict_conds.append(conflict_delinq)
            
        if not conflict_conds:
            return master_df.with_columns(pl.lit(False).alias("servicer_conflict_flag"))
            
        final_conflict = conflict_conds[0]
        for cond in conflict_conds[1:]:
            final_conflict = final_conflict | cond
            
        reconciled_df = joined.with_columns(
            final_conflict.alias("servicer_conflict_flag")
        )
        
        conflicts_count = reconciled_df["servicer_conflict_flag"].sum()
        logger.warning(f"Reconciliation complete. Flagged {conflicts_count} loans for 'Requires Manual Review'.")
        
        return reconciled_df

    def validate_logical_breaks(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        3. Cross-Column Logic Validation.
        Flags deterministic and impossible scenarios across columns.
        """
        logger.info("Validating logical breaks...")
        
        has_balance_cols = "current_balance" in df.columns and "original_balance" in df.columns
        has_date_cols = "reporting_month" in df.columns and "origination_month" in df.columns
        has_delinq_cols = "days_past_due" in df.columns and "delinquency_status" in df.columns

        bal_break = (
            (pl.col("current_balance").is_not_null()) & 
            (pl.col("original_balance").is_not_null()) & 
            (pl.col("current_balance") > pl.col("original_balance"))
        ) if has_balance_cols else pl.lit(False)
        
        date_break = (
            (pl.col("reporting_month").is_not_null()) & 
            (pl.col("origination_month").is_not_null()) & 
            (pl.col("reporting_month") < pl.col("origination_month"))
        ) if has_date_cols else pl.lit(False)
        
        delinq_break = (
            (pl.col("days_past_due").is_not_null()) & 
            (pl.col("delinquency_status").is_not_null()) & 
            (pl.col("days_past_due") > 30) & 
            (pl.col("delinquency_status") == "Current")
        ) if has_delinq_cols else pl.lit(False)
        
        validated_df = df.with_columns([
            bal_break.alias("break_balance_logic"),
            date_break.alias("break_date_logic"),
            delinq_break.alias("break_delinquency_logic")
        ])
        
        validated_df = validated_df.with_columns(
            (pl.col("break_balance_logic") | 
             pl.col("break_date_logic") | 
             pl.col("break_delinquency_logic")).alias("has_logical_breaks")
        )
        
        total_breaks = validated_df["has_logical_breaks"].sum()
        logger.info(f"Logical validation complete. Found {total_breaks} rows with breaks.")
        
        return validated_df

    def calculate_drift(self, train_df: pl.DataFrame, test_df: pl.DataFrame, columns: List[str], bins: int = 10) -> Dict[str, float]:
        """
        4. Population Stability Index (PSI) & Drift.
        Calculates PSI between training and test datasets.
        """
        logger.info("Calculating Population Stability Index (PSI) for data drift...")
        psi_scores = {}
        
        for col in columns:
            if col not in train_df.columns or col not in test_df.columns:
                continue
                
            train_col = train_df.select(col).drop_nulls().to_series()
            test_col = test_df.select(col).drop_nulls().to_series()
            
            if len(train_col) == 0 or len(test_col) == 0:
                psi_scores[col] = float('nan')
                continue
                
            if train_col.dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                min_val = min(train_col.min(), test_col.min())
                max_val = max(train_col.max(), test_col.max())
                
                if min_val == max_val:
                    psi_scores[col] = 0.0
                    continue
                    
                bins_edges = np.linspace(min_val, max_val, bins + 1)
                
                train_hist, _ = np.histogram(train_col.to_numpy(), bins=bins_edges)
                test_hist, _ = np.histogram(test_col.to_numpy(), bins=bins_edges)
                
                train_pct = train_hist / sum(train_hist)
                test_pct = test_hist / sum(test_hist)
                
            else:
                train_counts = train_df.group_by(col).agg(pl.len().alias("count"))
                test_counts = test_df.group_by(col).agg(pl.len().alias("count"))
                
                train_dict = dict(zip(train_counts[col].to_list(), train_counts["count"].to_list()))
                test_dict = dict(zip(test_counts[col].to_list(), test_counts["count"].to_list()))
                
                all_cats = set(train_dict.keys()).union(set(test_dict.keys()))
                
                train_len = train_df.height
                test_len = test_df.height
                
                train_pct = np.array([train_dict.get(c, 0) / train_len for c in all_cats])
                test_pct = np.array([test_dict.get(c, 0) / test_len for c in all_cats])
                
            # Avoid divide by zero
            train_pct = np.where(train_pct == 0, 0.0001, train_pct)
            test_pct = np.where(test_pct == 0, 0.0001, test_pct)
            
            psi = np.sum((train_pct - test_pct) * np.log(train_pct / test_pct))
            psi_scores[col] = psi
            
        logger.info("PSI Calculation complete.")
        return psi_scores

    def generate_dq_score(self, df: pl.DataFrame, critical_cols: List[str]) -> pl.DataFrame:
        """
        5. Record-Level Data Quality Score.
        Generates a Verifiability Score (0-100). Deducts points for breaks/conflicts.
        Outputs a Verification Report mapping.
        """
        logger.info("Generating record-level Data Quality (DQ) Scores...")
        
        scores = pl.Series("dq_score", [100] * df.height)
        df_scored = df.with_columns(scores)
        
        for col in critical_cols:
            if col in df_scored.columns:
                df_scored = df_scored.with_columns(
                    pl.when(pl.col(col).is_null())
                    .then(pl.col("dq_score") - 10)
                    .otherwise(pl.col("dq_score"))
                    .alias("dq_score")
                )
                
        if "has_logical_breaks" in df_scored.columns:
            df_scored = df_scored.with_columns(
                pl.when(pl.col("has_logical_breaks"))
                .then(pl.col("dq_score") - 25)
                .otherwise(pl.col("dq_score"))
                .alias("dq_score")
            )
            
        if "servicer_conflict_flag" in df_scored.columns:
            df_scored = df_scored.with_columns(
                pl.when(pl.col("servicer_conflict_flag"))
                .then(pl.col("dq_score") - 20)
                .otherwise(pl.col("dq_score"))
                .alias("dq_score")
            )
            
        # Ensure score boundaries
        df_scored = df_scored.with_columns(
            pl.when(pl.col("dq_score") < 0).then(0)
            .otherwise(pl.col("dq_score"))
            .alias("dq_score")
        )
        
        # Classification mapping
        df_scored = df_scored.with_columns(
            pl.when(pl.col("dq_score") >= 90).then(pl.lit("Auto-Approved"))
            .when(pl.col("dq_score") >= 70).then(pl.lit("Requires Manual Review"))
            .otherwise(pl.lit("Rejected"))
            .alias("verification_status")
        )
        
        logger.info("DQ Scores generated and verification statuses assigned.")
        return df_scored

    def export_pool_fact_sheet(self, df: pl.DataFrame, output_path: str = "reports/pool_fact_sheet.html"):
        """
        6. Visual Exports.
        Generates a 'Pool Fact Sheet' HTML report containing heatmaps, box plots, and pie charts.
        """
        logger.info(f"Generating Pool Fact Sheet HTML report at {output_path}...")
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "heatmap"}, {"type": "pie"}],
                   [{"type": "box"}, {"type": "bar"}]],
            subplot_titles=(
                "Missingness Heatmap", 
                "Risk Concentration (State)", 
                "Outliers: Current Balance", 
                "DQ Verification Status"
            )
        )
        
        # 1. Missingness Heatmap
        sample_cols = df.columns[:15]
        missing_matrix = df.select([pl.col(c).is_null().cast(pl.Int32) for c in sample_cols]).to_numpy()
        if missing_matrix.shape[0] > 1000:
            missing_matrix = missing_matrix[:1000, :]
            
        fig.add_trace(go.Heatmap(
            z=missing_matrix,
            x=sample_cols,
            colorscale='Blues',
            showscale=False
        ), row=1, col=1)
        
        # 2. Risk Concentration (State) - Pie Chart
        if "state" in df.columns:
            state_counts = df.group_by("state").agg(pl.len().alias("count")).to_pandas()
            fig.add_trace(go.Pie(
                labels=state_counts["state"], 
                values=state_counts["count"],
                hole=0.4
            ), row=1, col=2)
            
        # 3. Outlier Distributions (Box Plot using IQR for Current Balance)
        if "current_balance" in df.columns:
            bal_data = df.select("current_balance").drop_nulls().to_series().to_numpy()
            if len(bal_data) > 5000:
                bal_data = np.random.choice(bal_data, 5000, replace=False)
            fig.add_trace(go.Box(
                y=bal_data,
                name="Current Balance",
                boxpoints='outliers'
            ), row=2, col=1)
            
        # 4. DQ Verification Status - Bar Chart
        if "verification_status" in df.columns:
            status_counts = df.group_by("verification_status").agg(pl.len().alias("count")).to_pandas()
            
            # Map colors manually
            color_map = {"Auto-Approved": "green", "Requires Manual Review": "orange", "Rejected": "red"}
            colors = [color_map.get(s, "gray") for s in status_counts["verification_status"]]
            
            fig.add_trace(go.Bar(
                x=status_counts["verification_status"],
                y=status_counts["count"],
                marker_color=colors
            ), row=2, col=2)
            
        fig.update_layout(height=800, width=1200, title_text="Intain-Sight: Pool Fact Sheet")
        fig.write_html(output_path)
        logger.info(f"Pool Fact Sheet successfully exported to {output_path}")
