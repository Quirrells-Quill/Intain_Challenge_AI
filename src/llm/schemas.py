"""
Pydantic schemas for structured reviewer outputs — schemas.py
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class ReviewerSummarySchema(BaseModel):
    summary: str = Field(description="Concise 2-3 sentence executive synthesis of loan performance.")
    risk_assessment: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(description="Synthesized categorical risk level.")
    key_drivers: List[str] = Field(description="Top 3 to 5 quantitative drivers identified by SHAP or rule engines.")
    recommended_action: Literal["Auto-Approve", "Manual Triage", "Reject/Repurchase"] = Field(description="Recommended human reviewer triage action.")
    reviewer_notes: str = Field(description="Specific investigative instructions for the underwriting audit team.")
    grounding_citations: List[str] = Field(description="List of exact sections cited from data_dictionary or validation_rules.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Self-reported LLM confidence based on context sufficiency.")
    disclaimer: str = Field(default="LLM Advisory Recommendation: Not a binding underwriting decision.")

class DictionaryQuerySchema(BaseModel):
    field_name: str
    definition: str
    business_context: str
    validation_constraints: Optional[str]
    citations: List[str]
