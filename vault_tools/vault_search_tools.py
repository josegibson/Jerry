"""
Search-related tools and schemas for the vault.
Includes web search, semantic search, file listing, and content search.
"""

import os
import subprocess
from typing import List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from ._utils import (
    get_vault_path,
    check_tool_available,
    list_files_python,
    search_content_python,
)
from .vectorstore import get_retriever

# Allowed file-extensions for the vault listing/searching tools. Keep lower-case
# and without leading dots.
ALLOWED_EXTENSIONS = {
    "md",
    "txt",
    "json",
    "csv",
    "rst",
}


class VaultListInput(BaseModel):
    """Input schema for listing vault files."""

    ext: Optional[str] = Field(
        default="md", description="File extension to filter by (e.g., 'md', 'txt')"
    )
    hidden: bool = Field(
        default=False, description="Whether to include hidden files in the results"
    )

    @field_validator("ext", mode="before")
    def normalise_and_validate_ext(cls, v):
        """Ensure extension is one of the allowed set; strip leading dot."""
        if v is None or v == "":
            return "md"
        v = v.lower().lstrip(".")
        if v not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported extension '{v}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            )
        return v


class VaultSearchInput(BaseModel):
    """Input schema for searching vault content."""

    term: str = Field(description="The search term to look for in vault files")
    ignore_case: bool = Field(
        default=False, description="Whether to perform case-insensitive search"
    )
    max_count: Optional[int] = Field(
        default=None, description="Maximum number of matches per file"
    )

    @field_validator("term")
    def validate_term(cls, v):
        if not v or not v.strip():
            raise ValueError("Search term cannot be empty")
        if len(v.strip()) > 200:
            raise ValueError("Search term is too long (max 200 characters)")
        return v.strip()


class SemanticSearchInput(BaseModel):
    query: str = Field(description="Semantic search query against the vault index")
    k: int = Field(default=5, ge=1, le=10, description="Number of results")

    @field_validator("query")
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


@tool(args_schema=SemanticSearchInput)
def semantic_vault_search(query: str, k: int = 5) -> str:
    """Semantic search over the Obsidian vault vector store and return top-k chunks with sources."""
    try:
        retriever = get_retriever(
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", ".chroma"), top_k=k
        )
        docs = retriever.invoke(query)
        lines: List[str] = []
        for idx, doc in enumerate(docs, start=1):
            source = (
                doc.metadata.get("source")
                or doc.metadata.get("path")
                or doc.metadata.get("file_path")
                or "unknown"
            )
            lines.append(f"[{idx}] {source}\n{doc.page_content}")
        return "\n\n".join(lines) if lines else "No semantic matches"
    except Exception as e:
        return f"Semantic search error: {e}"


class WebSearchInput(BaseModel):
    query: str = Field(description="Web search query")
    max_results: int = Field(default=5, ge=1, le=10, description="Max results to return")

    @field_validator("query")
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


@tool(args_schema=WebSearchInput)
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return a concise set of results with titles and URLs.

    Uses Tavily if TAVILY_API_KEY is set; otherwise falls back to DuckDuckGo instant answers.
    """
    try:
        from tavily import TavilyClient  # type: ignore
        import os as _os

        api_key = _os.getenv("TAVILY_API_KEY")
        if api_key:
            client = TavilyClient(api_key=api_key)
            res = client.search(query=query, max_results=max_results)
            items = res.get("results", [])
            lines = []
            for item in items[:max_results]:
                title = item.get("title") or ""
                url = item.get("url") or ""
                snippet = item.get("content") or item.get("snippet") or ""
                lines.append(f"- {title}\n  {url}\n  {snippet}")
            return "\n".join(lines) if lines else "No results"
    except Exception:
        pass

    # Fallback to DuckDuckGo via requests (no API key)
    try:
        import requests

        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": 1, "no_html": 1},
            timeout=10,
        )
        data = resp.json()
        lines: List[str] = []
        for topic in data.get("RelatedTopics", [])[: max_results]:
            if isinstance(topic, dict):
                text = topic.get("Text") or ""
                url = topic.get("FirstURL") or ""
                if text or url:
                    lines.append(f"- {text}\n  {url}")
        return "\n".join(lines) if lines else "No results"
    except Exception as e:
        return f"Web search error: {e}"


@tool(args_schema=VaultListInput)
def vault_list_files(ext: str = "md", hidden: bool = False) -> str:
    """
    List files in the vault directory.

    Uses `fd` when available, otherwise falls back to a pure-Python traversal.
    """
    # Prefer fd if available for speed
    if check_tool_available("fd"):
        try:
            vault_path = get_vault_path()
        except ValueError as e:
            return f"Error: {e}"

        cmd = ["fd"]
        if ext:
            cmd.extend(["-e", ext])
        if hidden:
            cmd.append("--hidden")
        cmd.extend([".", str(vault_path)])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            file_list = result.stdout.strip()
            if not file_list:
                return f"No {ext} files found in vault"
            return file_list
        except subprocess.CalledProcessError as e:
            return f"Error running fd: {e.stderr}"

    # Fallback to Python
    try:
        files = list_files_python(ext, hidden)
        return "\n".join(files) if files else f"No {ext} files found in vault"
    except Exception as e:
        return f"Error listing files: {e}"


@tool(args_schema=VaultSearchInput)
def vault_search_content(
    term: str, ignore_case: bool = False, max_count: Optional[int] = None
) -> str:
    """
    Search for text content within vault files.

    Uses `ripgrep` when available, otherwise falls back to a pure-Python search.
    """
    if check_tool_available("rg"):
        try:
            vault_path = get_vault_path()
        except ValueError as e:
            return f"Error: {e}"

        cmd = ["rg", "--line-number", "--with-filename"]
        if ignore_case:
            cmd.append("--ignore-case")
        if max_count:
            cmd.extend(["--max-count", str(max_count)])
        cmd.extend(["--type", "md"])  # md files only
        cmd.append(term)
        cmd.append(str(vault_path))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return (
                result.stdout.strip() if result.stdout else f"No matches found for '{term}'"
            )
        except subprocess.CalledProcessError as e:
            if e.returncode == 1:
                return f"No matches found for '{term}'"
            return f"Error running ripgrep: {e.stderr}"

    # Fallback to Python
    try:
        matches = search_content_python(
            term, ignore_case=ignore_case, max_count=max_count, file_ext="md"
        )
        return "\n".join(matches) if matches else f"No matches found for '{term}'"
    except Exception as e:
        return f"Error searching content: {e}" 
    
SEARCH_TOOLS = [
    web_search,
    semantic_vault_search,
    vault_list_files,
    vault_search_content,
]