import importlib
import json
from pathlib import Path
from typing import Dict, Any

from core.monitoring.dev_monitor import DevMonitor
from core.agent.agent_system import AgentSystem


class Agent:
	"""Marker class retained for typing; real base is AgentSystem."""
	def __init__(self, context: Any):
		self.context = context


class SimpleContext:
	"""Simple container that exposes raw system instances as attributes."""
	def __init__(self, systems: Dict[str, Any]):
		for system_name, system_instance in systems.items():
			setattr(self, system_name, system_instance)


class AgentAssembler:
	"""
	Builds agents by discovering systems and wiring per-agent dependencies.
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
		"""Lazily initializes and returns a singleton system instance."""
		if system_name in self._initialized_systems:
			return self._initialized_systems[system_name]

		if system_name not in self.systems_registry:
			raise ValueError(f"System '{system_name}' not found in registry.")

		system_meta = self.systems_registry[system_name]
		module_path = system_meta["module_path"]
		class_name = system_meta["class_name"]

		try:
			module = importlib.import_module(module_path)
			system_class = getattr(module, class_name)
			instance = system_class(monitor=self.monitor)
			self._initialized_systems[system_name] = instance
			return instance
		except (ImportError, AttributeError) as e:
			self.monitor.log_event("runtime_error", "system_init_failed", {"system": system_name, "error": str(e)})
			raise RuntimeError(f"Could not initialize system '{system_name}': {e}")

	def load_agent(self, agent_name: str):
		"""
		Loads an agent by reading its manifest and injecting raw system instances.
		"""
		if agent_name in self.agents:
			raise ValueError(f"Agent '{agent_name}' is already loaded.")

		manifest_path = self.agents_dir / agent_name / "manifest.json"
		if not manifest_path.is_file():
			raise FileNotFoundError(f"Manifest for agent '{agent_name}' not found at {manifest_path}")

		with open(manifest_path, 'r') as f:
			agent_manifest = json.load(f)

		# Prepare agent-specific directories
		agent_dir = self.agents_dir / agent_name
		agent_data_dir = agent_dir / "data"
		agent_data_dir.mkdir(exist_ok=True)

		# Initialize systems
		required_systems = agent_manifest.get("capabilities_required", [])
		agent_systems = {}
		for sys_name in required_systems:
			system_meta = self.systems_registry[sys_name]
			module_path = system_meta["module_path"]
			class_name = system_meta["class_name"]
			module = importlib.import_module(module_path)
			system_class = getattr(module, class_name)

			if sys_name == "database_system":
				instance = system_class(monitor=self.monitor, db_path=str(agent_data_dir / "system.db"))
			elif sys_name == "file_system":
				instance = system_class(monitor=self.monitor, root_dir=agent_dir)
			elif sys_name == "planner_system":
				# Per-agent planner data stored under the agent's data directory
				instance = system_class(planner_data_path=agent_data_dir / "planner.json")
			else:
				instance = self._get_system_instance(sys_name)
			agent_systems[sys_name] = instance
		
		# Build a simple context with raw systems and data_dir hint
		agent_systems["data_dir"] = agent_data_dir
		agent_context = SimpleContext(agent_systems)
		# Expose declared capabilities on the context for runtime checks
		setattr(agent_context, "capabilities", set(required_systems))
		
		# Pass the context to the agent's constructor
		# If the agent inherits AgentSystem, it will manage its own entries.db
		agent_instance = AgentSystem(agent_context)
		# Attach manifest and capabilities to the agent instance for easy access
		setattr(agent_instance, "manifest", agent_manifest)
		setattr(agent_instance, "capabilities", set(required_systems))
		# Convenience helper for capability checks
		if not hasattr(agent_instance, "has_capability"):
			setattr(agent_instance, "has_capability", lambda name: name in getattr(agent_instance, "capabilities", set()))
		
		self.agents[agent_name] = agent_instance
		self.monitor.log_event("runtime", "agent_loaded", {"agent_name": agent_name})
		print(f"Agent '{agent_name}' loaded with capabilities: {list(agent_systems.keys())}")
		return agent_instance

	def get_agent(self, agent_name: str) -> Agent:
		"""Retrieves a loaded agent by name."""
		return self.agents.get(agent_name)

	def shutdown(self):
		"""Performs shutdown tasks for the agent assembler."""
		self.monitor.log_event("runtime", "shutdown", {})
		print("Agent assembler shutting down.")


