from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Any, List, Tuple, Dict

from .agent_monitor import AgentMonitor

# --- Core LangChain/DB Imports ---
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

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
# 1. VECTOR STORE MANAGER CLASS
# ==============================================================================

class VectorStoreManager:
    """
    Manages all vector store operations for a single, sandboxed agent.
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
            self.monitor.log_event("vector_store_debug", {"message": f"Selected OpenAI embeddings: {model}"})
            return OpenAIEmbeddings(model=model), f"OpenAI:{model}"
        if _HAS_HF_EMB:
            model = os.getenv("SENTENCE_TRANSFORMERS_MODEL", "all-MiniLM-L6-v2")
            self.monitor.log_event("vector_store_debug", {"message": f"Selected HuggingFace embeddings: {model}"})
            return HuggingFaceEmbeddings(model_name=model), f"HuggingFace:{model}"
            
        raise RuntimeError("No embedding backend available. Install langchain-openai or langchain-huggingface and set API keys.")

    def add_documents_from_path(self, source_path: Path, glob_pattern: str = "**/*.md") -> Dict[str, int]:
        """
        Loads, splits, and indexes all documents from a given path.
        Returns a dictionary with statistics about the indexing process.
        """
        self.monitor.log_event("vector_store_debug", {"message": f"Indexing documents from '{source_path}'"})
        
        docs: List[Document] = []
        
        for file_path in source_path.rglob(glob_pattern):
            if file_path.is_file():
                try:
                    loader = TextLoader(str(file_path), encoding="utf-8")
                    docs.extend(loader.load())
                except Exception as e:
                    self.monitor.log_event("vector_store_warning", {"message": f"Could not load file {file_path}. Error: {e}"})
        
        if not docs:
            self.monitor.log_event("vector_store_debug", {"message": "No new documents found to index."})
            return {"indexed_chunks": 0, "file_count": 0}

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        ids = [
            hashlib.sha256((chunk.page_content + chunk.metadata["source"]).encode()).hexdigest()
            for chunk in chunks
        ]
        
        self.db.add_documents(chunks, ids=ids)
        
        stats = {
            "indexed_chunks": len(chunks),
            "file_count": len(docs)
        }
        
        self.monitor.log_event("vector_store_info", {
            "message": f"Indexed {stats['indexed_chunks']} chunks from {stats['file_count']} files.",
            **stats
        })
        
        return stats

    def get_retriever(self, top_k: int = 5) -> Any:
        """Returns a LangChain retriever for this agent's vector store."""
        return self.db.as_retriever(search_kwargs={"k": top_k})

# ==============================================================================
# 2. PUBLIC FACTORY FUNCTION
# ==============================================================================

def get_vector_store_manager(persist_directory: Path, collection_name: str, monitor: AgentMonitor) -> VectorStoreManager:
    """
    Factory function to create and return a VectorStoreManager instance.
    """
    return VectorStoreManager(persist_directory, collection_name, monitor)
