from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

from langchain_community.document_loaders import ObsidianLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Prefer OpenAI embeddings if key present; otherwise use default all-MiniLM via SentenceTransformersEmbeddings
try:
    from langchain_openai import OpenAIEmbeddings
    _HAS_OPENAI_EMB = True
except Exception:
    _HAS_OPENAI_EMB = False

try:
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    _HAS_ST_EMB = True
except Exception:
    _HAS_ST_EMB = False

import chromadb
from chromadb.utils import embedding_functions
import hashlib

from ._utils import get_vault_path


@dataclass
class IndexStats:
    files_indexed: int
    chunks_indexed: int
    persist_directory: str
    embedding_model: str


def _select_embeddings() -> Tuple[Any, str]:
    """Select an embeddings backend based on env and availability."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if _HAS_OPENAI_EMB and openai_key:
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model, api_key=openai_key), f"OpenAI:{model}"
    if _HAS_ST_EMB:
        model = os.getenv("SENTENCE_TRANSFORMERS_MODEL", "all-MiniLM-L6-v2")
        return SentenceTransformerEmbeddings(model_name=model), f"ST:{model}"
    raise RuntimeError("No embeddings backend available. Install langchain-openai or sentence-transformers.")


def _load_obsidian_docs(vault_path: Path, glob: str = "**/*.md") -> List[Document]:
    loader = ObsidianLoader(str(vault_path))
    return loader.load()


def _split_docs(documents: List[Document], chunk_size: int = 1200, chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["---", "\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)


def ensure_vector_index(
    persist_directory: str = ".chroma",
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    file_glob: str = "**/*.md",
) -> Dict[str, Any]:
    """Build or update a Chroma vector index from the Obsidian vault."""
    vault_path = get_vault_path()
    embeddings, emb_name = _select_embeddings()
    persist_dir = Path(persist_directory)
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Load and chunk
    documents = _load_obsidian_docs(vault_path, glob=file_glob)
    chunks = _split_docs(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Upsert into Chroma
    vs = Chroma(
        embedding_function=embeddings,
        collection_name=os.getenv("CHROMA_COLLECTION", "jerry_vault"),
        persist_directory=str(persist_dir),
    )
    if chunks:
        ids = [hashlib.md5(chunk.page_content.encode() + chunk.metadata["source"].encode() + str(i).encode()).hexdigest() for i, chunk in enumerate(chunks)]
        vs.add_documents(chunks, ids=ids)

    return {
        "_stats": IndexStats(
            files_indexed=len(documents),
            chunks_indexed=len(chunks),
            persist_directory=str(persist_dir),
            embedding_model=emb_name,
        ).__dict__
    }


# -----------------------------
# LangChain Retriever (for agent)
# -----------------------------
def get_retriever(
    persist_directory: str = ".chroma",
    top_k: int = 5,
) -> Any:
    """Return a retriever over the persisted Chroma collection (LangChain style)."""
    embeddings, _ = _select_embeddings()
    vs = Chroma(
        embedding_function=embeddings,
        collection_name=os.getenv("CHROMA_COLLECTION", "jerry_vault"),
        persist_directory=persist_directory,
    )
    return vs.as_retriever(search_kwargs={"k": top_k})


# -----------------------------
# Direct Chroma Access (for CLI/manual tools)
# -----------------------------
class _EmbeddingWrapper(embedding_functions.EmbeddingFunction):
    """Wrapper to adapt LangChain embeddings to Chroma's native API."""
    def __init__(self, embed_model):
        self.embed_model = embed_model

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embed_model.embed_documents(input)


def get_chroma_collection(
    persist_directory: str = ".chroma",
    collection_name: str = None,
) -> Any:
    """Return a direct Chroma collection without LangChain."""
    if not collection_name:
        collection_name = os.getenv("CHROMA_COLLECTION", "jerry_vault")

    embeddings, _ = _select_embeddings()
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=_EmbeddingWrapper(embeddings)
    )
    return collection


def semantic_search_direct(
    query: str,
    k: int = 5,
    persist_directory: str = ".chroma",
) -> List[Dict[str, Any]]:
    """Direct Chroma semantic search without LangChain."""
    collection = get_chroma_collection(persist_directory)
    results = collection.query(query_texts=[query], n_results=k)
    output = []
    for doc, meta, score in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        output.append({"source": meta.get("source", "Unknown"), "content": doc, "score": score})
    return output
