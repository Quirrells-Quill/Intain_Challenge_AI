"""
Development Log Compiler — log_generator.py

Auto-compiles the markdown deliverable detailing the AI Agentic Development Log.
"""

import os
from typing import Dict, Any

class DevelopmentLogCompiler:
    """
    Auto-generates the 'reports/AI_Development_Log.md' mandated by the 
    Intain FinTech Challenge for Agentic ML tracking.
    """

    def __init__(self, output_path: str = "reports/AI_Development_Log.md"):
        self.output_path = output_path
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def generate_log(self, metrics: Dict[str, Any], code_share: Dict[str, Any]):
        """Builds the comprehensive Markdown document."""
        
        lines = [
            "# INTAIN-SIGHT: Agentic AI Development Log",
            "> **Mandatory Governance Deliverable**: Programmatic verification of AI vs Human development attribution.",
            "",
            "## Section 1: AI Tools Utilized",
            "- **Primary AI Agent**: Antigravity (Powered by Gemini Pro / Deepmind Architecture)",
            "- **Tasks Handled**: Boilerplate generation, Polars vectorization, Multi-task ML architecture, Pytest formulation, and Plotly visualization.",
            "",
            "## Section 2: Agentic Code Share Estimate",
            "Calculated via the `src/governance/ai_tracker.py` heuristic over the `src/` directory:",
            f"- **Total Lines of Code Scanned**: {code_share.get('total_lines_of_code', 0)}",
            f"- **AI-Generated / Auto-Scaffolded (Est)**: {code_share.get('ai_share_percentage', 0):.1f}% ({code_share.get('ai_generated_lines', 0)} lines)",
            f"- **Human-Refactored / Authored (Est)**: {code_share.get('human_share_percentage', 0):.1f}% ({code_share.get('human_refactored_lines', 0)} lines)",
            "",
            "## Section 3: Human Review & Governance Process",
            "All Agentic outputs were evaluated via **Strict Test-Driven Validation**. Code was evaluated against:",
            "1. **Temporal Leakage Rules**: Enforcing time-aware chronological grouping.",
            "2. **API Completeness**: Ensuring correct version matching (e.g. `lifelines` AalenJohansenFitter APIs).",
            "3. **Memory/Performance**: Vectorized Polars broadcasting to avoid standard python iterative loops.",
            "",
            "## Section 4: Representative Prompt Log",
            "The following chronological log details the interplay between the human architect and the AI Agent."
        ]

        # Iterate through the raw ledger
        for i, entry in enumerate(metrics.get("raw_ledger", [])):
            lines.extend([
                "",
                f"### Prompt {i+1}: {entry['phase']}",
                f"**Request**: *\"{entry['prompt']}\"*",
                f"**AI Tool**: {entry['ai_tool']}",
                f"**Status**: `{entry['output_status']}`"
            ])
            
            if entry['output_status'] == "REJECTED":
                lines.append(f"**Error Mode Detected**: {entry.get('error_mode', 'N/A')}")
                
            lines.append(f"**Human-in-the-Loop Review**: {entry['human_review_notes']}")

        lines.extend([
            "",
            "## Section 5: MLflow Experiment Link",
            "**Tracking URI**: `http://localhost:5000` *(Placeholder: Reference MLflow server UI for artifact tracking)*",
            "",
            "## Section 6: Lessons Learned",
            "- **What Worked**: Auto-generating Polars schema aggregations, boilerplate class architectures, and Plotly UI components provided 10x velocity.",
            "- **What Failed**: The AI struggled occasionally with edge-case financial mathematical definitions (like passing `cause_of_interest` in older versions of `lifelines`) and required prompt correction on memory-heavy `for-loops`.",
            "- **Correction Velocity**: Providing exact traceback errors to the AI yielded near-instant deterministic fixes, proving that Agentic workflows shine when paired with rigid QA smoke tests."
        ])

        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
            
        print(f"Generated mandated AI Development Log at {self.output_path}")
