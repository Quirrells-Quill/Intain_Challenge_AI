"""
Anomaly Audit Report Generator — report_generator.py

Compiles a reviewer-ready dossier of the top 20 highest-confidence anomaly
cases, stratified across exception types for coverage balance:
    - ≥5 Data Logic Errors
    - ≥5 Servicer Discrepancies
    - ≥10 Multivariate ML Outliers

Each case receives a structured verification dossier with a natural-language
audit narrative generated from the observed feature values and exception context.

Outputs:
    reports/ANOMALY_REVIEWER_DOSSIER.md  — Markdown report for audit sign-off
    reports/ANOMALY_REVIEWER_DOSSIER.html — HTML table for system integration
"""

import polars as pl
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from src.utils.logger import get_logger

logger = get_logger(__name__)

MD_OUTPUT = Path("reports/ANOMALY_REVIEWER_DOSSIER.md")
HTML_OUTPUT = Path("reports/ANOMALY_REVIEWER_DOSSIER.html")

# Stratified case quotas per exception type
QUOTAS: Dict[str, int] = {
    "Data Logic Error":       5,
    "Servicer Discrepancy":   5,
    "Severe Deterioration":   10,
}


class AnomalyAuditReportGenerator:
    """
    Extracts top anomaly cases and generates structured audit dossiers
    in Markdown and HTML format for compliance review.
    """

    def __init__(self):
        MD_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        logger.info("AnomalyAuditReportGenerator initialized.")

    # ------------------------------------------------------------------
    # Case Selection
    # ------------------------------------------------------------------

    def select_cases(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Selects up to 20 diverse anomaly cases using stratified quota sampling.

        Quotas ensure at least 5 Data Logic Errors and 5 Servicer Discrepancies
        are represented, with ML outliers filling remaining slots.

        Args:
            df: Full anomaly-scored DataFrame with exception_type column.

        Returns:
            pl.DataFrame: Up to 20 cases sorted by anomaly_score descending.
        """
        if "exception_type" not in df.columns or "anomaly_score" not in df.columns:
            raise ValueError("DataFrame must contain 'exception_type' and 'anomaly_score'.")

        cases: List[pl.DataFrame] = []

        for ex_type, quota in QUOTAS.items():
            subset = (
                df.filter(pl.col("exception_type") == ex_type)
                .sort("anomaly_score", descending=True)
                .head(quota)
            )
            if subset.height > 0:
                cases.append(subset)
                logger.info(f"  [{ex_type}]: {subset.height} cases selected.")

        if not cases:
            logger.warning("No exception cases found. Returning top 20 by anomaly_score.")
            return df.sort("anomaly_score", descending=True).head(20)

        combined = pl.concat(cases).unique(subset=["loan_id", "reporting_month"])
        combined = combined.sort("anomaly_score", descending=True).head(20)
        logger.info(f"Total cases selected for dossier: {combined.height}")
        return combined

    # ------------------------------------------------------------------
    # Narrative Generation
    # ------------------------------------------------------------------

    def generate_narrative(self, row: Dict) -> str:
        """
        Produces a natural language audit narrative for a single anomaly case.

        The narrative is structured to be immediately actionable by a human
        reviewer, citing specific values, conflict types, and recommended steps.

        Args:
            row: Dictionary of column values for one anomaly record.

        Returns:
            str: Plain-English audit narrative (1–3 sentences).
        """
        loan_id = row.get("loan_id", "Unknown")
        score = row.get("anomaly_score", 0.0)
        ex_type = row.get("exception_type", "None")
        action = row.get("recommended_action", "Auto-Approve")
        drivers = row.get("top_drivers", "None")
        state = row.get("state", "N/A")
        servicer = row.get("servicer_name", "Unknown Servicer")
        current_bal = row.get("current_balance", "N/A")
        orig_bal = row.get("original_balance", "N/A")
        dpd = row.get("days_past_due", "N/A")
        recon_notes = row.get("reconciliation_notes", "")

        driver_list = drivers.replace(";", ", ") if drivers and drivers != "None" else "multiple risk indicators"

        if ex_type == "Data Logic Error":
            return (
                f"Loan {loan_id} ({state}) flagged for '{action}' with anomaly score {score:.1f}/100 "
                f"due to deterministic accounting violations: {driver_list}. "
                f"Current balance reported as ${current_bal:,.2f} against an original balance of "
                f"${orig_bal:,.2f}, violating balance integrity constraints. "
                f"Immediate data correction and servicer confirmation required before pool reporting."
                if isinstance(current_bal, (int, float)) and isinstance(orig_bal, (int, float))
                else
                f"Loan {loan_id} ({state}) flagged for '{action}' with anomaly score {score:.1f}/100 "
                f"due to deterministic rule violations ({driver_list}). "
                f"Record must be corrected by data operations before inclusion in pool summary."
            )

        elif ex_type == "Servicer Discrepancy":
            return (
                f"Loan {loan_id} ({state}) flagged for '{action}' (score: {score:.1f}/100) "
                f"due to a data conflict between {servicer} and the master record. "
                f"Reconciliation note: '{recon_notes}'. "
                f"Top contributing factors: {driver_list}. "
                f"Servicer feed must be reconciled and a corrected tape submitted within 5 business days."
            )

        else:
            return (
                f"Loan {loan_id} ({state}) identified as a multivariate ML outlier "
                f"(score: {score:.1f}/100, action: '{action}'). "
                f"The unsupervised ensemble detected abnormal patterns in: {driver_list}. "
                f"Days past due: {dpd}. This loan's risk profile deviates significantly from "
                f"the training population distribution — manual underwriting review recommended."
            )

    # ------------------------------------------------------------------
    # Report Export
    # ------------------------------------------------------------------

    def generate_report(self, df: pl.DataFrame) -> str:
        """
        Selects top cases, generates narratives, and exports Markdown + HTML reports.

        Args:
            df: Full anomaly-scored DataFrame.

        Returns:
            str: Path to the generated Markdown report.
        """
        cases = self.select_cases(df)
        rows = cases.to_dicts()

        md_lines: List[str] = [
            "# INTAIN-SIGHT: ANOMALY REVIEWER DOSSIER",
            f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
            f"Total Cases: {len(rows)}  |  Reviewer Sign-Off Required",
            "",
            "---",
            "",
        ]

        html_rows: List[str] = []

        for i, row in enumerate(rows, start=1):
            narrative = self.generate_narrative(row)
            ex_type = row.get("exception_type", "None")

            # Map exception type to severity badge color
            badge_color = {
                "Data Logic Error": "🔴",
                "Servicer Discrepancy": "🟠",
                "Severe Deterioration": "🔴",
                "None": "🟢",
            }.get(ex_type, "🟡")

            # ── Markdown dossier entry ────────────────────────────────────
            md_lines += [
                f"## Case {i:02d} {badge_color} — Loan `{row.get('loan_id', 'N/A')}`",
                "",
                "| Field                | Value |",
                "|----------------------|-------|",
                f"| **Loan ID**          | `{row.get('loan_id', 'N/A')}` |",
                f"| **Reporting Month**  | {row.get('reporting_month', 'N/A')} |",
                f"| **State**            | {row.get('state', 'N/A')} |",
                f"| **Vintage**          | {row.get('origination_vintage', 'N/A')} |",
                f"| **Servicer**         | {row.get('servicer_name', 'N/A')} |",
                f"| **Anomaly Score**    | **{row.get('anomaly_score', 0.0):.1f} / 100** |",
                f"| **Exception Type**   | {ex_type} |",
                f"| **Recommended Action** | **{row.get('recommended_action', 'N/A')}** |",
                f"| **Top Drivers**      | `{row.get('top_drivers', 'None')}` |",
                f"| **Days Past Due**    | {row.get('days_past_due', 'N/A')} |",
                f"| **Current Balance**  | {row.get('current_balance', 'N/A')} |",
                "",
                f"**Audit Narrative:** {narrative}",
                "",
                "---",
                "",
            ]

            # ── HTML table row ────────────────────────────────────────────
            action_color = {
                "Reject/Repurchase": "#ff4444",
                "Manual Triage": "#ff8800",
                "Auto-Approve": "#44aa44",
            }.get(row.get("recommended_action", ""), "#888888")

            html_rows.append(
                f"<tr>"
                f"<td>{i}</td>"
                f"<td><code>{row.get('loan_id', 'N/A')}</code></td>"
                f"<td>{row.get('reporting_month', 'N/A')}</td>"
                f"<td>{row.get('state', 'N/A')}</td>"
                f"<td><b>{row.get('anomaly_score', 0.0):.1f}</b></td>"
                f"<td>{ex_type}</td>"
                f"<td style='color:{action_color};font-weight:bold'>{row.get('recommended_action', 'N/A')}</td>"
                f"<td><small>{row.get('top_drivers', 'None')}</small></td>"
                f"<td><small>{narrative[:120]}...</small></td>"
                f"</tr>"
            )

        # Write Markdown
        md_text = "\n".join(md_lines)
        MD_OUTPUT.write_text(md_text, encoding="utf-8")
        logger.info(f"Markdown dossier written → {MD_OUTPUT}")

        # Write HTML
        html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Intain-Sight Anomaly Reviewer Dossier</title>
  <style>
    body {{ font-family: Arial, sans-serif; font-size: 13px; padding: 20px; background: #f9f9f9; }}
    h1 {{ color: #1a3a5c; }}
    table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
    th {{ background: #1a3a5c; color: white; padding: 10px; text-align: left; }}
    td {{ padding: 8px 10px; border-bottom: 1px solid #ddd; vertical-align: top; }}
    tr:hover {{ background: #f0f4ff; }}
  </style>
</head>
<body>
  <h1>&#128221; Intain-Sight: Anomaly Reviewer Dossier</h1>
  <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {len(rows)} cases</p>
  <table>
    <thead>
      <tr>
        <th>#</th><th>Loan ID</th><th>Month</th><th>State</th>
        <th>Score</th><th>Exception Type</th><th>Action</th>
        <th>Top Drivers</th><th>Audit Narrative</th>
      </tr>
    </thead>
    <tbody>
      {"".join(html_rows)}
    </tbody>
  </table>
</body>
</html>"""
        HTML_OUTPUT.write_text(html_content, encoding="utf-8")
        logger.info(f"HTML dossier written → {HTML_OUTPUT}")

        return str(MD_OUTPUT)
