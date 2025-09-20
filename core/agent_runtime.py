import json
from pathlib import Path
from typing import Dict, Any, Optional, List

# --- Core LangChain Imports ---
from langchain_core.load import dumpd, load

# --- Project Imports ---
from .graph_builder import create_agent_graph, ConversationState
from .tool_provider import ToolProvider
# CHANGE: Import the new manager factory instead of the old retriever function
from .vectorstore_manager import get_vector_store_manager


class AgentRuntime:
    """
    Manages the entire lifecycle of a directory-scoped agent, acting as the
    central controller that assembles all necessary components.
    """

    def __init__(self, root_dir: str, name: Optional[str] = None,
                 provider: str = "gemini", system_prompt: str = None):
        
        self.root_dir = Path(root_dir).resolve()
        self.config_path = self.root_dir / "agent.json"
        self.state_path = self.root_dir / ".agent_state.json"
        self.workspace_dir = self.root_dir / "workspace"
        self.chroma_dir = self.root_dir / ".chroma"

        if not self.root_dir.exists():
            self._create_new_agent_scaffold(name=name or self.root_dir.name,
                                 provider=provider,
                                 system_prompt=system_prompt)

        # Load configuration and persistent state
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = json.load(f)
        self._load_state()

        # --- NEW: Instantiate the agent's dedicated VectorStoreManager ---
        self.vector_store_manager = get_vector_store_manager(
            persist_directory=self.chroma_dir,
            collection_name=self.config.get("name", "default_agent_collection")
        )
        
        # --- NEW: Index documents from the workspace on startup ---
        self.vector_store_manager.add_documents_from_path(self.workspace_dir)

        # Assemble and build the agent's graph on initialization
        self.graph = self._initialize_graph()


    def _initialize_graph(self):
        """
        Assembles all agent-specific components and uses the factory function
        to build a ready-to-use LangGraph instance.
        """
        print(f"[Init] Building graph for agent '{self.config.get('name')}'...")
        
        provider = ToolProvider(
            workspace_path=self.workspace_dir,
            retriever=self.get_retriever()
        )
        
        requested_tools = self.config.get("tools", [])
        agent_tools = provider.get_tools(requested_tools)

        provider_pref = self.config.get("provider")
        system_prompt = self.config.get("system_prompt", "You are a helpful assistant.")

        return create_agent_graph(
            tools=agent_tools,
            retriever=self.get_retriever(),
            system_prompt=system_prompt,
            preferred_provider=provider_pref
        )

    def _load_state(self):
        """Loads and deserializes the agent's persistent state from JSON."""
        if self.state_path.exists():
            with open(self.state_path, "r", encoding="utf-8") as f:
                state_data = json.load(f)
                messages_json = state_data.get("messages", [])
                self.state: ConversationState = {**state_data}
                self.state["messages"] = [load(m) for m in messages_json]
        else:
            self.state: ConversationState = {"messages": []}
            self.save_state()

    def save_state(self):
        """Serializes and persists the agent's state to JSON."""
        state_to_save = {**self.state}
        state_to_save["messages"] = [dumpd(msg) for msg in self.state["messages"]]
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state_to_save, f, indent=2)

    def get_retriever(self):
        """
        Returns a vectorstore retriever configured for this agent's Chroma DB
        by calling the agent's own VectorStoreManager.
        """
        # CHANGE: This is now a simple pass-through call.
        return self.vector_store_manager.get_retriever(
            top_k=int(self.config.get("retrieve_top_k", 5))
        )
        
    def _create_new_agent_scaffold(self, name: str, provider: str, system_prompt: Optional[str]):
        """Creates the necessary files and directories for a new agent."""
        print(f"[Init] Scaffolding new agent at: {self.root_dir}")
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(exist_ok=True)
        self.chroma_dir.mkdir(exist_ok=True)

        config = {
            "name": name,
            "provider": provider,
            "system_prompt": system_prompt or "You are a helpful assistant.",
            "retrieve_top_k": 5,
            "tools": ["file_tools", "knowledge_tools"] # Use tool group names
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        initial_state = {"messages": []}
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(initial_state, f, indent=2)

