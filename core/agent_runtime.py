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
from .agent_monitor import AgentMonitor


from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from datetime import datetime

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
        self.log_dir = self.root_dir / "logs"

        if not self.root_dir.exists():
            self._create_new_agent_scaffold(name=name or self.root_dir.name,
                                 provider=provider,
                                 system_prompt=system_prompt)

        # Load configuration and persistent state
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = json.load(f)
        self._load_state()

        # --- Instantiate monitoring and logging --- 
        self.log_dir.mkdir(exist_ok=True)
        self.monitor = AgentMonitor(agent_name=self.config.get("name", "unnamed_agent"), log_dir=self.log_dir)
        self.monitor.log_event("agent_startup", {"config": self.config})

        # --- Instantiate the VectorStoreManager ---
        self.vector_store_manager = get_vector_store_manager(
            persist_directory=self.chroma_dir,
            collection_name=self.config.get("name", "default_agent_collection"),
            monitor=self.monitor
        )
        
        # --- Index documents on startup ---
        self.vector_store_manager.add_documents_from_path(self.workspace_dir)

        # --- Assemble the agent's graph ---
        self.graph = self._initialize_graph()

    # ==========================================================================
    # --- PUBLIC API METHODS ---
    # ==========================================================================

    def invoke(self, input_text: str) -> Dict[str, Any]:
        """
        Handles a single turn of the conversation.
        """
        self.state["messages"].append(HumanMessage(content=input_text))
        
        final_state = self.graph.invoke(self.state)
        self.state = final_state
        self.save_state()
        
        last_ai_message = next((m for m in reversed(final_state.get("messages", [])) if isinstance(m, AIMessage)), None)
        response_text = last_ai_message.content if last_ai_message else "No response."
        
        return {
            "response": response_text,
            "tool_results": final_state.get("tool_results")
        }

    def analyze_knowledge_base(self) -> Dict[str, Any]:
        """
        Performs and returns an analysis of the agent's knowledge base.
        """
        return self.monitor.get_knowledge_base_analysis(self.workspace_dir, self.vector_store_manager)

    def shutdown(self) -> Dict[str, Any]:
        """
        Performs all shutdown tasks for the agent.
        """
        self.monitor.log_event("agent_shutdown_start", {})
        self._archive_session_as_markdown()
        self._clear_persistent_state()
        
        final_metrics = self.monitor.get_token_metrics()
        self.monitor.log_event("agent_shutdown_complete", {"final_token_metrics": final_metrics})
        
        return {
            "message": "Session archived, state cleared, and final metrics logged.",
            "final_metrics": final_metrics
        }

    def get_config(self) -> Dict[str, Any]:
        """Returns the agent's configuration dictionary."""
        return self.config

    def reindex_workspace(self) -> Dict[str, int]:
        """Rescans the workspace and updates the knowledge base."""
        return self.vector_store_manager.add_documents_from_path(self.workspace_dir)

    def get_session_history(self) -> str:
        """Returns a formatted string of the current session's history."""
        if not self.state.get("messages"):
            return "No messages in this session yet."
        
        history_str = "" 
        for msg in self.state["messages"]:
            if isinstance(msg, HumanMessage):
                history_str += f"\n--- You ---\n{msg.content}"
            elif isinstance(msg, AIMessage):
                history_str += f"\n--- Agent ---\n{msg.content}"
            elif isinstance(msg, ToolMessage):
                history_str += f"\n--- Tool Output ({msg.tool_call_id}) ---\n{msg.content}"
        return history_str

    # ==========================================================================
    # --- INTERNAL & SETUP METHODS ---
    # ==========================================================================

    def _clear_persistent_state(self):
        """Resets the persistent state file to an empty state."""
        initial_state = {"messages": []}
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(initial_state, f, indent=2)
        self.monitor.log_event("state_cleared", {"path": str(self.state_path)})
    
    def _initialize_graph(self):
        """Assembles the LangGraph instance."""
        self.monitor.log_event("graph_build_start", {"agent_name": self.config.get('name')})
        
        provider = ToolProvider(
            workspace_path=self.workspace_dir,
            retriever=self.get_retriever(),
            monitor=self.monitor
        )
        
        requested_tools = self.config.get("tools", [])
        agent_tools = provider.get_tools(requested_tools)

        provider_pref = self.config.get("provider")
        system_prompt = self.config.get("system_prompt", "You are a helpful assistant.")

        return create_agent_graph(
            tools=agent_tools,
            retriever=self.get_retriever(),
            system_prompt=system_prompt,
            monitor=self.monitor,
            preferred_provider=provider_pref
        )

    def _load_state(self):
        """Loads the agent's persistent state from JSON."""
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

    def _archive_session_as_markdown(self):
        """Formats the current session's history into a Markdown file."""
        sessions_dir = self.workspace_dir / "sessions"
        sessions_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = sessions_dir / f"{timestamp}.md"
        
        transcript = f"# Agent Session: {timestamp}\n\n"
        
        for msg in self.state.get("messages", []):
            if isinstance(msg, HumanMessage):
                transcript += f"**👤 You:**\n```text\n{msg.content}\n```\n\n"
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    transcript += f"**🤖 {self.config.get('name')}:**\n"
                    for tc in msg.tool_calls:
                        args = json.dumps(tc['args'], indent=2)
                        transcript += f"- `Tool Call: {tc['name']}`\n  ```json\n{args}\n  ```\n"
                else:
                    transcript += f"**🤖 {self.config.get('name')}:**\n```text\n{msg.content}\n```\n\n"
            elif isinstance(msg, ToolMessage):
                transcript += f"**🔧 Tool Output (`{msg.tool_call_id}`):**\n```text\n{msg.content}\n```\n\n"
        
        archive_path.write_text(transcript, encoding="utf-8")
        self.monitor.log_event("session_archived", {"path": str(archive_path)})

    def get_retriever(self):
        """Returns a vectorstore retriever for this agent."""
        return self.vector_store_manager.get_retriever(
            top_k=int(self.config.get("retrieve_top_k", 5))
        )
        
    def _create_new_agent_scaffold(self, name: str, provider: str, system_prompt: Optional[str]):
        """Creates the necessary files and directories for a new agent."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(exist_ok=True)
        self.chroma_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        config = {
            "name": name,
            "provider": provider,
            "system_prompt": system_prompt or "You are a helpful assistant.",
            "retrieve_top_k": 5,
            "tools": ["file_tools", "knowledge_tools"]
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        initial_state = {"messages": []}
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(initial_state, f, indent=2)

