from __future__ import annotations

from pathlib import Path
from typing import Any, List

from langchain_core.tools import BaseTool, tool # Import 'tool' decorator

from .agent_monitor import AgentMonitor

# --- Import the factory functions directly ---
from core.tools.file_tools import create_file_tools
from core.tools.knowledge_tools import create_knowledge_tools

# --- Stateless tools can be imported as they are ---
from core.tools.web_search_tools import web_search

# --- Import Planner Module and its tools ---
from planner.planner_module import PlannerModule # Assuming planner is at project root
from core.tools import planner_tools # Import the module

class ToolProvider:
    """
    A factory that constructs and provides a sandboxed, context-aware
    set of tools for a specific agent instance.
    """

    def __init__(self, workspace_path: Path, retriever: Any, monitor: AgentMonitor):
        """
        Initializes the provider with the agent's unique context.

        Args:
            workspace_path: The root directory for the agent's file operations.
            retriever: The agent's configured LangChain retriever instance.
            monitor: The agent's monitor instance for logging.
        """
        self.workspace_path = workspace_path
        self.retriever = retriever
        self.monitor = monitor

        # --- Initialize Planner Module and its tools ---
        # The planner_data.json will be in the top-level 'planner' directory
        planner_data_path = self.workspace_path.parent.parent / "planner" / "planner_data.json"
        planner_data_path.parent.mkdir(parents=True, exist_ok=True) # Ensure planner directory exists
        self.planner_module = PlannerModule(planner_data_path)
        planner_tools.initialize_planner_tools(self.planner_module)

        # Helper to wrap a function as a LangChain tool
        def _wrap_as_tool(func):
            return tool(func)

        self._tool_registry = {
            "file_tools": lambda: create_file_tools(self.workspace_path),
            "knowledge_tools": lambda: create_knowledge_tools(self.retriever),
            "web_search_tools": lambda: [web_search],
            "planner_tools": lambda: [
                _wrap_as_tool(planner_tools.add_planner_task),
                _wrap_as_tool(planner_tools.get_planner_due_tasks),
                _wrap_as_tool(planner_tools.mark_planner_task_completed),
                _wrap_as_tool(planner_tools.mark_planner_task_failed),
                _wrap_as_tool(planner_tools.mark_planner_task_cancelled),
                _wrap_as_tool(planner_tools.update_planner_task),
            ],
        }

    def get_tools(self, requested_tools: List[str]) -> List[BaseTool]:
        """
        Builds and returns a list of tool instances based on agent configuration.
        """
        final_tools: List[BaseTool] = []
        for tool_name in requested_tools:
            if tool_name in self._tool_registry:
                tool_factory = self._tool_registry[tool_name]
                final_tools.extend(tool_factory())
            else:
                self.monitor.log_event("tool_provider_warning", {
                    "message": f"Requested tool group '{tool_name}' not found in registry."
                })

        self.monitor.log_event("tool_provider_info", {
            "message": "Built tools for agent.",
            "tool_names": [tool.name for tool in final_tools]
        })
        return final_tools
