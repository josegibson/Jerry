from __future__ import annotations

import json
from typing import Any, List

from langchain_core.tools import BaseTool, tool


def create_knowledge_tools(retriever: Any) -> List[BaseTool]:
    """
    Factory function to create a tool for searching the agent's knowledge base.

    Args:
        retriever: The LangChain retriever instance configured for the agent's vector store.

    Returns:
        A list containing the configured knowledge base search tool.
    """

    @tool
    def search_knowledge_base(query: str) -> str:
        """
        Performs a semantic search for information related to the query in the
        agent's indexed knowledge base.
        """
        try:
            docs = retriever.invoke(query)
            results = [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "content": doc.page_content,
                }
                for doc in docs
            ]
            return json.dumps(results, indent=2)
        except Exception as e:
            return f'{{"error": "SearchError", "message": "{str(e)}"}}'

    return [search_knowledge_base]
