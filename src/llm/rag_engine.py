"""
RAG Engine — rag_engine.py

Grounded Retrieval-Augmented Generation using Gemini embeddings 
and normalized cosine similarity for in-memory retrieval.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from src.utils.logger import get_logger
from src.llm.gemini_client import GovernedGeminiClient

logger = get_logger(__name__)

def cosine_sim(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

class LoanKnowledgeRAG:
    """
    In-memory vector store for Data Dictionary and Validation Rules.
    Chunks documents and retrieves using Gemini embeddings.
    """

    def __init__(self, client: GovernedGeminiClient, dict_path: str = "data/data_dictionary.md", rules_path: str = "configs/validation_rules.json"):
        self.client = client
        self.dict_path = Path(dict_path)
        self.rules_path = Path(rules_path)
        self.documents = []
        
        self._load_documents()

    def _load_documents(self):
        """Loads and chunks documents, embedding them into memory."""
        logger.info("Initializing LoanKnowledgeRAG: Chunking and Embedding...")
        doc_id = 0
        raw_docs = []
        
        # Load Data Dictionary
        if self.dict_path.exists():
            with open(self.dict_path, 'r', encoding='utf-8') as f:
                content = f.read()
                chunks = content.split('##')
                for chunk in chunks:
                    if len(chunk.strip()) > 10:
                        header = chunk.split('\n')[0].strip()
                        text = chunk.strip()
                        raw_docs.append({
                            "id": f"dict_sec_{doc_id}", 
                            "source": f"data_dictionary.md (Section: {header})", 
                            "text": text
                        })
                        doc_id += 1
                        
        # Load Validation Rules
        if self.rules_path.exists():
            with open(self.rules_path, 'r', encoding='utf-8') as f:
                try:
                    rules = json.load(f)
                    for category, ruleset in rules.items():
                        text = f"Category: {category}\nRules: {json.dumps(ruleset)}"
                        raw_docs.append({
                            "id": f"rule_sec_{doc_id}", 
                            "source": f"validation_rules.json (Category: {category})", 
                            "text": text
                        })
                        doc_id += 1
                except json.JSONDecodeError:
                    logger.warning("Failed to parse validation_rules.json")

        # Embed all docs
        for doc in raw_docs:
            emb = self.client.get_embeddings(doc["text"])
            doc["embedding"] = emb
            self.documents.append(doc)
            
        logger.info(f"RAG Engine successfully indexed {len(self.documents)} document chunks.")

    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieves exact dictionary definitions and rules with source citation tags.
        """
        if not self.documents:
            return []
            
        query_emb = self.client.get_embeddings(query)
        
        # Score all documents
        scored_docs = []
        for doc in self.documents:
            score = cosine_sim(query_emb, doc["embedding"])
            scored_docs.append((score, doc))
            
        # Sort descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for i in range(min(top_k, len(scored_docs))):
            score, doc = scored_docs[i]
            # Simple thresholding
            if score > 0.4 or self.client.mock_mode:
                results.append({
                    "score": score,
                    "source": doc["source"],
                    "text": doc["text"]
                })
                
        return results
