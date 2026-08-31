"""
Google Gemini API wrapper with structured JSON enforcement — gemini_client.py
"""

import os
import json
import google.generativeai as genai
from pydantic import BaseModel
from typing import Type, TypeVar, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T', bound=BaseModel)

class GovernedGeminiClient:
    """
    Wraps the Gemini API to enforce structured JSON generation mapped to Pydantic schemas.
    Provides graceful offline mock fallback for CI environments.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.mock_mode = not bool(self.api_key) or self.api_key == "MOCK_KEY_FOR_SMOKE_TEST"
        self.model_name = model_name
        
        self.system_instruction = (
            "You are the Intain Verification Agent AI Copilot. You assist human reviewers "
            "in evaluating loan-level data anomalies and performance risks. You do NOT make final "
            "credit decisions. Every factual assertion must be grounded in the provided ML output "
            "or retrieved documentation. You must return only valid JSON complying with the requested schema."
        )

        if not self.mock_mode:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.system_instruction
            )
            logger.info(f"GovernedGeminiClient initialized with active API key (model: {model_name})")
        else:
            logger.warning("GEMINI_API_KEY not found. Running GovernedGeminiClient in MOCK MODE.")

    def generate_structured(self, prompt: str, schema_class: Type[T]) -> T:
        """
        Calls Gemini, requests JSON matching the Pydantic schema, and validates the output.
        """
        if self.mock_mode:
            return self._mock_response(schema_class)

        schema_json = schema_class.model_json_schema()
        full_prompt = (
            f"{prompt}\n\n"
            f"You MUST output valid JSON only, conforming exactly to this JSON schema:\n"
            f"{json.dumps(schema_json, indent=2)}\n"
            f"Do not include markdown code blocks, just raw JSON."
        )

        try:
            config = genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
            response = self.model.generate_content(full_prompt, generation_config=config)
            
            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            
            validated_obj = schema_class.model_validate_json(raw_text.strip())
            return validated_obj
            
        except Exception as e:
            logger.error(f"Failed to generate structured LLM output: {e}")
            raise

    def get_embeddings(self, text: str) -> list[float]:
        """Fetches embeddings using Gemini models/text-embedding-004."""
        if self.mock_mode:
            # Return dummy embedding of length 768
            return [0.01] * 768
            
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return [0.0] * 768

    def _mock_response(self, schema_class: Type[T]) -> T:
        """Generates deterministic mock responses for CI."""
        if schema_class.__name__ == "ReviewerSummarySchema":
            return schema_class.model_validate({
                "summary": "Mock summary: Loan shows 18.2% default probability.",
                "risk_assessment": "MEDIUM",
                "key_drivers": ["interest_rate", "dti"],
                "recommended_action": "Manual Triage",
                "reviewer_notes": "Mock note.",
                "grounding_citations": ["Rule 1"],
                "confidence_score": 0.85,
                "disclaimer": "LLM Advisory Recommendation: Not a binding underwriting decision."
            })
        elif schema_class.__name__ == "DictionaryQuerySchema":
            return schema_class.model_validate({
                "field_name": "Mock Field",
                "definition": "Mock definition.",
                "business_context": "Mock context.",
                "validation_constraints": None,
                "citations": ["Source A"]
            })
        else:
            raise ValueError(f"Unknown mock schema: {schema_class.__name__}")
