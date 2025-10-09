from pathlib import Path
from typing import Dict, Type

# --- System Imports ---
# from core.database.database_system import DatabaseSystem
# from core.vector_db.vector_db_system import VectorDBSystem
# from core.planner.planner_system import PlannerSystem
# from core.file_system.file_system import FileSystem
# from core.event_bus.event_bus import EventBus
from .agent_monitor import AgentMonitor

class Agent:
    """Base class for all agents."""
    def __init__(self, agent_runtime):
        self.runtime = agent_runtime

class AgentRuntime:
    """
    Manages the lifecycle of the entire multi-agent system.
    """

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        self.log_dir = self.root_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # --- Initialize Systems ---
        self.monitor = AgentMonitor(agent_name="system", log_dir=self.log_dir)
        # self.db = DatabaseSystem(db_path=str(self.root_dir / "jerry.db"))
        # self.vdb = VectorDBSystem(persist_directory=self.root_dir / ".chroma", collection_name="jerry_vdb", monitor=self.monitor)
        # self.planner = PlannerSystem(planner_data_path=self.root_dir / "planner_data.json")
        # self.fs = FileSystem(workspace_path=self.root_dir / "workspace")
        # self.event_bus = EventBus()

        self.systems = {
            # "db": None,
            # "vdb": None,
            # "planner": None,
            # "fs": None,
            # "event_bus": None,
            "monitor": self.monitor
        }

        self.agents: Dict[str, Agent] = {}

    def load_agent(self, agent_name: str, agent_class: Type[Agent]):
        """
        Loads and initializes an agent, injecting system dependencies.
        """
        if agent_name in self.agents:
            raise ValueError(f"Agent with name '{agent_name}' is already loaded.")
        
        agent_instance = agent_class(self)
        self.agents[agent_name] = agent_instance
        self.monitor.log_event("agent_loaded", {"agent_name": agent_name})
        print(f"Agent '{agent_name}' loaded.")
        return agent_instance

    def get_agent(self, agent_name: str) -> Agent:
        """Retrieves a loaded agent by name."""
        return self.agents.get(agent_name)

    def shutdown(self):
        """Performs shutdown tasks for the agent runtime."""
        self.monitor.log_event("runtime_shutdown", {})
        print("Agent runtime shutting down.")
