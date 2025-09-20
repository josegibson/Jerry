from __future__ import annotations

from pathlib import Path
from typing import Any, List

from langchain_core.tools import BaseTool

# --- Import the factory functions directly ---
# This is cleaner than importing classes that are no longer needed.
from core.tools.file_tools import create_file_tools
from core.tools.knowledge_tools import create_knowledge_tools

# --- Stateless tools can be imported as they are ---
from core.tools.web_search_tools import web_search


class ToolProvider:
    """
    A factory that constructs and provides a sandboxed, context-aware
    set of tools for a specific agent instance.
    """

    def __init__(self, workspace_path: Path, retriever: Any):
        """
        Initializes the provider with the agent's unique context.

        Args:
            workspace_path: The root directory for the agent's file operations.
            retriever: The agent's configured LangChain retriever instance.
        """
        self.workspace_path = workspace_path
        self.retriever = retriever

        # The registry now maps tool group names to their corresponding factory
        # functions. We use lambdas to defer the actual creation of the tools
        # until they are requested by the agent.
        self._tool_registry = {
            "file_tools": lambda: create_file_tools(self.workspace_path),
            "knowledge_tools": lambda: create_knowledge_tools(self.retriever),
            "web_search_tools": lambda: [web_search],  # Stateless tools are simply returned in a list.
        }

    def get_tools(self, requested_tools: List[str]) -> List[BaseTool]:
        """
        Builds and returns a list of tool instances based on agent configuration.

        This method iterates through the requested tool groups, calls the
        appropriate factory function from the registry, and aggregates the results.

        Args:
            requested_tools: A list of tool group names from the agent's config file.
        """
        final_tools: List[BaseTool] = []
        for tool_name in requested_tools:
            if tool_name in self._tool_registry:
                # Call the factory function to get the list of tools.
                tool_factory = self._tool_registry[tool_name]
                final_tools.extend(tool_factory())
            else:
                print(f"[Provider] Warning: Requested tool group '{tool_name}' not found in registry.")

        print(f"[Provider] Built tools: {[tool.name for tool in final_tools]}")
        return final_tools
