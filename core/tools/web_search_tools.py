from __future__ import annotations

from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """Searches the web."""
    # Your web search implementation would go here.
    return "Stateless web search results."