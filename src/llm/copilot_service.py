"""
Copilot Service — copilot_service.py

High-level Reviewer Copilot API integrating Gemini, RAG, Guardrails, and Audit Logging.
"""

import json
import time
from typing import Dict, Any, Optional
from src.utils.logger import get_logger

from src.llm.gemini_client import GovernedGeminiClient
from src.llm.rag_engine import LoanKnowledgeRAG
from src.llm.guardrails import HallucinationGuardrail
from src.llm.audit_logger import AuditTrailLogger
from src.llm.schemas import ReviewerSummarySchema, DictionaryQuerySchema

logger = get_logger(__name__)


class ReviewerCopilotService:
    """
    Orchestrates the entire LLM Copilot workflow for a loan reviewer.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.client = GovernedGeminiClient(model_name=model_name)
        self.rag = LoanKnowledgeRAG(client=self.client)
        self.guardrails = HallucinationGuardrail()
        self.audit = AuditTrailLogger()
        logger.info("ReviewerCopilotService successfully initialized.")

    def generate_loan_dossier_note(
        self, 
        loan_id: str, 
        loan_data: Dict[str, Any], 
        ml_outputs: Dict[str, Any]
    ) -> ReviewerSummarySchema:
        """
        Retrieves relevant dictionary context via RAG, builds grounded prompt,
        calls Gemini, validates via guardrails, and logs transaction.
        """
        logger.info(f"Copilot generating dossier for loan: {loan_id}")
        start_time = time.time()
        
        # 1. Prepare Context & RAG Retrieval
        context_dict = {
            "loan_data": loan_data,
            "ml_outputs": ml_outputs
        }
        
        rag_context = ""
        # Search for drivers and anomaly scores
        top_drivers = ml_outputs.get("anomaly_data", {}).get("top_drivers", [])
        for driver in top_drivers:
            query = str(driver).replace("RULE_", "Rule ").replace("_", " ")
            retrieved = self.rag.retrieve_context(query, top_k=1)
            for r in retrieved:
                rag_context += f"From {r['source']}: {r['text']}\n"
                
        # 2. Build Prompt
        prompt = (
            f"Loan ID: {loan_id}\n\n"
            f"=== QUANTITATIVE DATA ===\n"
            f"{json.dumps(context_dict, indent=2)}\n\n"
            f"=== GOVERNANCE CONTEXT (RAG) ===\n"
            f"{rag_context}\n\n"
            f"Task: Synthesize a professional reviewer summary, identify risk drivers, and recommend an action.\n"
            f"Do not hallucinate numbers. Use exactly the probabilities provided."
        )

        # 3. Generate Structured Response
        raw_response = ""
        parsed_json = ""
        try:
            response_obj = self.client.generate_structured(prompt, ReviewerSummarySchema)
            parsed_json = response_obj.model_dump_json()
            raw_response = parsed_json # In this setup, raw and parsed represent the successful output path
        except Exception as e:
            logger.error(f"LLM Generation failed for loan {loan_id}: {e}")
            # Fallback on failure
            _, status, response_obj = self.guardrails._reject_and_fallback(ml_outputs, f"API/JSON Error: {e}")
            guardrail_status = "SCHEMA_ERROR"
            latency = time.time() - start_time
            self.audit.log_interaction(
                loan_id=loan_id, prompt_template=prompt, retrieved_context=rag_context,
                model_name=self.client.model_name, raw_response=str(e), parsed_json="{}",
                guardrail_status=guardrail_status, latency_seconds=latency
            )
            return response_obj

        # 4. Guardrail Verification
        is_valid, status, final_response = self.guardrails.validate_numerical_consistency(response_obj, ml_outputs)

        # 5. Audit Logging
        latency = time.time() - start_time
        self.audit.log_interaction(
            loan_id=loan_id,
            prompt_template=prompt,
            retrieved_context=rag_context,
            model_name=self.client.model_name,
            raw_response=raw_response,
            parsed_json=final_response.model_dump_json(),
            guardrail_status=status,
            latency_seconds=latency,
            human_reviewer_feedback="PENDING"
        )

        return final_response

    def explain_scenario_impact(self, scenario_name: str, simulation_results: Dict) -> str:
        """
        Synthesizes Monte Carlo simulation quantiles into an analyst briefing note.
        """
        prompt = (
            f"Scenario: {scenario_name}\n"
            f"Simulation Results: {json.dumps(simulation_results, indent=2)}\n"
            f"Synthesize this into a brief 2-paragraph analyst note detailing the stress impacts."
        )
        # For simplicity, bypassing Pydantic for raw text output in this method
        try:
            if self.client.mock_mode:
                return f"Mock Note: Scenario {scenario_name} shows significant stress."
            
            response = self.client.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating scenario impact: {e}"

    def query_dictionary(self, user_query: str) -> DictionaryQuerySchema:
        """
        Natural language Q&A over financial definitions.
        """
        retrieved = self.rag.retrieve_context(user_query, top_k=2)
        rag_context = "\n".join([r['text'] for r in retrieved])
        
        prompt = (
            f"Query: {user_query}\n\n"
            f"Context:\n{rag_context}\n\n"
            f"Extract the formal definition, business context, and constraints based ONLY on the context."
        )
        
        try:
            return self.client.generate_structured(prompt, DictionaryQuerySchema)
        except Exception:
            return DictionaryQuerySchema(
                field_name=user_query,
                definition="Retrieval failed.",
                business_context="",
                validation_constraints=None,
                citations=[]
            )
