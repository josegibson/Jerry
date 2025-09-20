from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Any, List, Tuple

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
# This class encapsulates all logic for a single agent's vector store.
# ==============================================================================

class VectorStoreManager:
    """
    Manages all vector store operations for a single, sandboxed agent.

    This includes selecting embedding models, loading and chunking documents
    from the agent's workspace, and creating a retriever instance.
    """
    def __init__(self, persist_directory: Path, collection_name: str):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function, self.embedding_model_name = self._select_embeddings()
        
        # Ensure the Chroma DB directory exists
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
            return OpenAIEmbeddings(model=model), f"OpenAI:{model}"
        if _HAS_HF_EMB:
            model = os.getenv("SENTENCE_TRANSFORMERS_MODEL", "all-MiniLM-L6-v2")
            # Use the modern, correct class
            return HuggingFaceEmbeddings(model_name=model), f"HuggingFace:{model}"
            
        raise RuntimeError("No embedding backend available. Install langchain-openai or langchain-huggingface and set API keys.")

    def add_documents_from_path(self, source_path: Path, glob_pattern: str = "**/*.md"):
        """
        Loads, splits, and indexes all documents from a given path into the
        agent's vector store.
        """
        print(f"[DB] Indexing documents from '{source_path}'...")
        docs: List[Document] = []
        for file_path in source_path.glob(glob_pattern):
            if file_path.is_file():
                try:
                    loader = TextLoader(str(file_path), encoding="utf-8")
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"[DB] Warning: Could not load file {file_path}. Error: {e}")
        
        if not docs:
            print("[DB] No new documents found to index.")
            return

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        # Generate unique, content-based IDs for idempotency
        ids = [
            hashlib.sha256((chunk.page_content + chunk.metadata["source"]).encode()).hexdigest()
            for chunk in chunks
        ]
        
        # Upsert chunks into the database
        self.db.add_documents(chunks, ids=ids)
        print(f"[DB] Indexed {len(chunks)} chunks from {len(docs)} files.")

    def get_retriever(self, top_k: int = 5) -> Any:
        """
        Returns a LangChain retriever configured for this agent's vector store.
        """
        return self.db.as_retriever(search_kwargs={"k": top_k})

# ==============================================================================
# 2. PUBLIC FACTORY FUNCTION (Optional but Recommended)
# This can be used by AgentRuntime to simplify retriever creation.
# ==============================================================================

def get_vector_store_manager(persist_directory: Path, collection_name: str) -> VectorStoreManager:
    """
    Factory function to create and return a VectorStoreManager instance.
    """
    return VectorStoreManager(persist_directory, collection_name)
