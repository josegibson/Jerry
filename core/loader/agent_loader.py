from pathlib import Path
from typing import Dict, Any

from core.monitoring.dev_monitor import DevMonitor
from core.agent.agent_system import AgentSystem
from .system_registry import SystemRegistry
from .manifest_parser import ManifestParser
from .dependency_injector import DependencyInjector


class SimpleContext:
    """Simple container that exposes raw system instances as attributes."""
    def __init__(self, systems: Dict[str, Any]):
        for system_name, system_instance in systems.items():
            setattr(self, system_name, system_instance)


class AgentLoader:
    """
    Main orchestrator for loading agents from directories.
    Handles system discovery, manifest parsing, and dependency injection.
    """
    
    def __init__(self, root_dir: str):
        """
        Initialize the agent loader.
        
        Args:
            root_dir: Root directory of the project
        """
        self.root_dir = Path(root_dir).resolve()
        self.log_dir = self.root_dir / "logs"
        self.systems_dir = self.root_dir / "core" / "system"
        self.agents_dir = self.root_dir / "agents"
        
        self.monitor = DevMonitor(log_dir=self.log_dir)
        
        # Initialize components
        self.system_registry = SystemRegistry(self.systems_dir, self.monitor)
        self.manifest_parser = ManifestParser(self.monitor)
        self.dependency_injector = DependencyInjector(self.monitor)
        
        self.agents: Dict[str, AgentSystem] = {}
    
    def load_agent(self, agent_name: str) -> AgentSystem:
        """
        Load an agent by reading its manifest and injecting dependencies.
        
        Args:
            agent_name: Name of the agent to load
            
        Returns:
            Loaded agent instance
            
        Raises:
            ValueError: If agent is already loaded
            FileNotFoundError: If agent manifest not found
        """
        if agent_name in self.agents:
            raise ValueError(f"Agent '{agent_name}' is already loaded.")
        
        # Parse agent manifest
        manifest_path = self.agents_dir / agent_name / "manifest.json"
        agent_manifest = self.manifest_parser.parse_agent_manifest(manifest_path)
        
        # Prepare agent-specific directories
        agent_dir = self.agents_dir / agent_name
        agent_data_dir = agent_dir / "data"
        agent_data_dir.mkdir(exist_ok=True)
        
        # Initialize required systems
        required_systems = agent_manifest.get("capabilities_required", [])
        agent_systems = {}
        
        for sys_name in required_systems:
            if not self.system_registry.is_system_available(sys_name):
                raise ValueError(f"Required system '{sys_name}' is not available.")
            
            system_manifest = self.system_registry.get_system_manifest(sys_name)
            system_instance = self.dependency_injector.create_system_instance(
                system_manifest, agent_data_dir, agent_dir
            )
            agent_systems[sys_name] = system_instance
        
        # Build agent context
        agent_systems["data_dir"] = agent_data_dir
        agent_context = SimpleContext(agent_systems)
        
        # Expose declared capabilities on the context
        setattr(agent_context, "capabilities", set(required_systems))
        
        # Create agent instance
        agent_instance = AgentSystem(agent_context)
        
        # Attach metadata to agent instance
        setattr(agent_instance, "manifest", agent_manifest)
        setattr(agent_instance, "capabilities", set(required_systems))
        
        # Add convenience helper for capability checks
        if not hasattr(agent_instance, "has_capability"):
            setattr(agent_instance, "has_capability", 
                   lambda name: name in getattr(agent_instance, "capabilities", set()))
        
        self.agents[agent_name] = agent_instance
        self.monitor.log_event("agent_loader", "agent_loaded", {"agent_name": agent_name})
        print(f"Agent '{agent_name}' loaded with capabilities: {list(agent_systems.keys())}")
        
        return agent_instance
    
    def get_agent(self, agent_name: str) -> AgentSystem:
        """
        Get a loaded agent by name.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent instance or None if not found
        """
        return self.agents.get(agent_name)
    
    def list_loaded_agents(self) -> list[str]:
        """Get a list of loaded agent names."""
        return list(self.agents.keys())
    
    def shutdown(self):
        """Perform shutdown tasks."""
        self.monitor.log_event("agent_loader", "shutdown", {})
        print("Agent loader shutting down.")

