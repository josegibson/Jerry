from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Any, List, Tuple, Dict

from core.agent.agent_monitor import AgentMonitor

# --- Core LangChain/DB Imports ---
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Embedding Model Imports ---
try:
    from langchain_openai import OpenAIEmbeddings
    _HAS_OPENAI_EMB = True
except ImportError:
    _HAS_OPENAI_EMB = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    _HAS_HF_EMB = True
except ImportError:
    _HAS_HF_EMB = False

# ==============================================================================
# 1. VECTOR DB SYSTEM CLASS
# ==============================================================================

class VectorDBSystem:
    """
    Manages all vector store operations, conforming to the system architecture.
    """
    def __init__(self, persist_directory: Path, collection_name: str, monitor: AgentMonitor):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.monitor = monitor
        self.embedding_function, self.embedding_model_name = self._select_embeddings()
        
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.db = Chroma(
            embedding_function=self.embedding_function,
            collection_name=self.collection_name,
            persist_directory=str(self.persist_directory),
        )

    def _select_embeddings(self) -> Tuple[Any, str]:
        """Selects an embedding model based on environment variables."""
        if _HAS_OPENAI_EMB and os.getenv("OPENAI_API_KEY"):
            model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
            self.monitor.log_event("vector_db_debug", {"message": f"Selected OpenAI embeddings: {model}"})
            return OpenAIEmbeddings(model=model), f"OpenAI:{model}"
        if _HAS_HF_EMB:
            model = os.getenv("SENTENCE_TRANSFORMERS_MODEL", "all-MiniLM-L6-v2")
            self.monitor.log_event("vector_db_debug", {"message": f"Selected HuggingFace embeddings: {model}"})
            return HuggingFaceEmbeddings(model_name=model), f"HuggingFace:{model}"
            
        raise RuntimeError("No embedding backend available. Install langchain-openai or langchain-huggingface and set API keys.")

    def addDocument(self, text_content: str, metadata: dict = None) -> str:
        """
        Adds a single document (text content) to the vector store.
        """
        metadata = metadata or {}
        doc = Document(page_content=text_content, metadata=metadata)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        chunks = text_splitter.split_documents([doc])
        
        doc_id = hashlib.sha256(text_content.encode()).hexdigest()
        ids = [f"{doc_id}-{i}" for i in range(len(chunks))]

        self.db.add_documents(chunks, ids=ids)
        self.monitor.log_event("vector_db_info", {"message": f"Added 1 document ({len(chunks)} chunks) with ID: {doc_id}"})
        return doc_id

    def semanticSearch(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a semantic search against the vector store.
        """
        results = self.db.similarity_search_with_score(query_text, k=top_k)
        
        formatted_results = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]
        self.monitor.log_event("vector_db_info", {"message": f"Performed semantic search for '{query_text[:50]}...'. Found {len(results)} results."})
        return formatted_results

