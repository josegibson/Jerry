from pathlib import Path
from typing import Dict, Type, Any

# --- System Imports ---
from ..dev_monitor.dev_monitor import DevMonitor
# I will recreate these systems one by one later
# from ..database.database_system import DatabaseSystem
# from ..event_bus.event_bus import EventBus
# ... etc

# =============================================================================
# --- Agent Context and Proxy Classes ---
# =============================================================================

class SystemProxy:
    """
    A proxy that sits between an agent and a real system.
    It automatically injects the agent's ID into every method call.
    """
    def __init__(self, agent_id: str, system: Any):
        self._agent_id = agent_id
        self._system = system

    def __getattr__(self, name: str) -> Any:
        # Get the real method from the system instance
        method = getattr(self._system, name)
        if not callable(method):
            return method

        # Return a new function that wraps the original method
        # and prepends the agent_id to the arguments.
        def wrapper(*args, **kwargs):
            return method(self._agent_id, *args, **kwargs)
        
        return wrapper

class AgentContext:
    """
    Provides an agent with a sandboxed, agent-aware interface to the systems.
    """
    def __init__(self, agent_id: str, systems: Dict[str, Any]):
        self.agent_id = agent_id
        
        # Create proxies for each system
        for system_name, system_instance in systems.items():
            if system_instance: # Only create proxies for implemented systems
                setattr(self, system_name, SystemProxy(agent_id, system_instance))

# =============================================================================
# --- Base Agent Class ---
# =============================================================================

class Agent:
    """Base class for all agents."""
    def __init__(self, context: AgentContext):
        self.context = context

# =============================================================================
# --- Agent Runtime Class ---
# =============================================================================

class AgentRuntime:
    """
    Manages the lifecycle of the entire multi-agent system.
    """
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        self.log_dir = self.root_dir / "logs"
        
        # --- Initialize Systems ---
        self.monitor = DevMonitor(log_dir=self.log_dir)
        
        # TODO: Recreate and uncomment these systems
        # self.db = DatabaseSystem(monitor=self.monitor, db_path=str(self.root_dir / "jerry.db"))
        # self.event_bus = EventBus(monitor=self.monitor)
        # ... etc.

        self.systems = {
            "db": None,
            "event_bus": None,
            "vdb": None,
            "planner": None,
            "fs": None,
            "monitor": self.monitor, # monitor is not proxied
        }

        self.agents: Dict[str, Agent] = {}

    def load_agent(self, agent_name: str, agent_class: Type[Agent]):
        """
        Loads and initializes an agent, injecting an AgentContext.
        """
        if agent_name in self.agents:
            raise ValueError(f"Agent with name '{agent_name}' is already loaded.")
        
        # Create a context for this specific agent
        agent_context = AgentContext(agent_name, self.systems)
        
        # Pass the context to the agent's constructor
        agent_instance = agent_class(agent_context)
        
        self.agents[agent_name] = agent_instance
        self.monitor.log_event("runtime", "agent_loaded", {"agent_name": agent_name})
        print(f"Agent '{agent_name}' loaded.")
        return agent_instance

    def get_agent(self, agent_name: str) -> Agent:
        """Retrieves a loaded agent by name."""
        return self.agents.get(agent_name)

    def shutdown(self):
        """Performs shutdown tasks for the agent runtime."""
        self.monitor.log_event("runtime", "shutdown", {})
        print("Agent runtime shutting down.")