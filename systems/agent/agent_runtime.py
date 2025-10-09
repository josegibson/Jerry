
import importlib
import json
from pathlib import Path
from typing import Dict, Type, Any, List

from systems.dev_monitor.dev_monitor import DevMonitor

# =============================================================================
# --- Agent Context and Proxy Classes (Largely Unchanged) ---
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
        method = getattr(self._system, name)
        if not callable(method):
            return method

        def wrapper(*args, **kwargs):
            return method(self._agent_id, *args, **kwargs)
        
        return wrapper

class AgentContext:
    """
    Provides an agent with a sandboxed, agent-aware interface to the systems.
    """
    def __init__(self, agent_id: str, systems: Dict[str, Any]):
        self.agent_id = agent_id
        for system_name, system_instance in systems.items():
            setattr(self, system_name, SystemProxy(agent_id, system_instance))

# =============================================================================
# --- Base Agent Class (Unchanged) ---
# =============================================================================

class Agent:
    """Base class for all agents."""
    def __init__(self, context: AgentContext):
        self.context = context

# =============================================================================
# --- New Agent Runtime Class ---
# =============================================================================

class AgentRuntime:
    """
    Manages the lifecycle of the multi-agent system using manifest-driven discovery.
    """
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        self.log_dir = self.root_dir / "logs"
        self.systems_dir = self.root_dir / "systems"
        self.agents_dir = self.root_dir / "agents"
        
        self.monitor = DevMonitor(log_dir=self.log_dir)
        
        self.systems_registry: Dict[str, Dict[str, Any]] = {}
        self.agents: Dict[str, Agent] = {}
        self._initialized_systems: Dict[str, Any] = {}

        self._discover_systems()

    def _discover_systems(self):
        """Scans the systems directory for manifests and populates the registry."""
        print("Discovering systems...")
        for system_dir in self.systems_dir.iterdir():
            manifest_path = system_dir / "manifest.json"
            if manifest_path.is_file():
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                        name = manifest.get("name")
                        if name:
                            print(f"  -> Found system: {name}")
                            self.systems_registry[name] = manifest
                except (IOError, json.JSONDecodeError) as e:
                    self.monitor.log_event("runtime_error", "system_discovery_failed", {"path": str(manifest_path), "error": str(e)})

    def _get_system_instance(self, system_name: str) -> Any:
        """Lazily initializes and returns a system instance."""
        # Return already initialized instance if available
        if system_name in self._initialized_systems:
            return self._initialized_systems[system_name]

        # Check if the system is in our registry
        if system_name not in self.systems_registry:
            raise ValueError(f"System '{system_name}' not found in registry.")

        # Dynamically import and initialize
        system_meta = self.systems_registry[system_name]
        module_path = system_meta["module_path"]
        class_name = system_meta["class_name"]

        try:
            module = importlib.import_module(module_path)
            system_class = getattr(module, class_name)
            
            # For now, we hardcode the dependencies for the known systems.
            if system_name == "database_system":
                agent_data_dir = self.root_dir / "_agent_data"
                agent_data_dir.mkdir(exist_ok=True)
                instance = system_class(monitor=self.monitor, db_path=str(agent_data_dir / "system.db"))
            elif system_name == "file_system":
                instance = system_class(monitor=self.monitor, root_dir=self.root_dir)
            else:
                instance = system_class(monitor=self.monitor)

            self._initialized_systems[system_name] = instance
            return instance
        except (ImportError, AttributeError) as e:
            self.monitor.log_event("runtime_error", "system_init_failed", {"system": system_name, "error": str(e)})
            raise RuntimeError(f"Could not initialize system '{system_name}': {e}")

    def load_agent(self, agent_name: str):
        """
        Loads an agent by reading its manifest, initializing required systems,
        and injecting them into an AgentContext.
        """
        if agent_name in self.agents:
            raise ValueError(f"Agent '{agent_name}' is already loaded.")

        manifest_path = self.agents_dir / agent_name / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest for agent '{agent_name}' not found at {manifest_path}")

        with open(manifest_path, 'r') as f:
            agent_manifest = json.load(f)

        # Initialize only the systems the agent requires
        required_systems = agent_manifest.get("capabilities_required", [])
        agent_systems = {sys_name: self._get_system_instance(sys_name) for sys_name in required_systems}
        
        # Add monitor by default, but don't proxy it
        # agent_systems["monitor"] = self.monitor 

        # Create a context for this specific agent
        agent_context = AgentContext(agent_name, agent_systems)
        
        # Dynamically load the agent class
        module_path = agent_manifest["module_path"]
        class_name = agent_manifest["class_name"]
        try:
            module = importlib.import_module(module_path)
            agent_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            self.monitor.log_event("runtime_error", "agent_load_failed", {"agent": agent_name, "error": str(e)})
            raise RuntimeError(f"Could not load agent class for '{agent_name}': {e}")

        # Pass the context to the agent's constructor
        agent_instance = agent_class(agent_context)
        
        self.agents[agent_name] = agent_instance
        self.monitor.log_event("runtime", "agent_loaded", {"agent_name": agent_name})
        print(f"Agent '{agent_name}' loaded with capabilities: {list(agent_systems.keys())}")
        return agent_instance

    def get_agent(self, agent_name: str) -> Agent:
        """Retrieves a loaded agent by name."""
        return self.agents.get(agent_name)

    def shutdown(self):
        """Performs shutdown tasks for the agent runtime."""
        self.monitor.log_event("runtime", "shutdown", {})
        print("Agent runtime shutting down.")
