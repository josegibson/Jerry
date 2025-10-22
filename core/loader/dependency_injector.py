import importlib
from pathlib import Path
from typing import Dict, Any

from core.monitoring.dev_monitor import DevMonitor


class DependencyInjector:
    """
    Handles dependency injection for agent systems.
    """
    
    def __init__(self, monitor: DevMonitor):
        """
        Initialize the dependency injector.
        
        Args:
            monitor: DevMonitor instance for logging
        """
        self.monitor = monitor
        self._initialized_systems: Dict[str, Any] = {}
    
    def create_system_instance(self, system_manifest: Dict[str, Any], 
                             agent_data_dir: Path, agent_dir: Path) -> Any:
        """
        Create a system instance based on its manifest.
        
        Args:
            system_manifest: The system's manifest dictionary
            agent_data_dir: Agent's data directory
            agent_dir: Agent's main directory
            
        Returns:
            Initialized system instance
        """
        system_name = system_manifest["name"]
        module_path = system_manifest["module_path"]
        class_name = system_manifest["class_name"]
        
        try:
            module = importlib.import_module(module_path)
            system_class = getattr(module, class_name)
            
            # Create instance with appropriate parameters based on system type
            if system_name == "database_system":
                instance = system_class(monitor=self.monitor, db_path=str(agent_data_dir / "system.db"))
            elif system_name == "file_system":
                instance = system_class(monitor=self.monitor, root_dir=agent_dir)
            elif system_name == "planner_system":
                instance = system_class(planner_data_path=agent_data_dir / "planner.json")
            else:
                instance = system_class(monitor=self.monitor)
            
            self.monitor.log_event("dependency_injector", "system_created", 
                                 {"system_name": system_name, "class_name": class_name})
            return instance
            
        except (ImportError, AttributeError) as e:
            self.monitor.log_event("dependency_injector", "system_creation_failed", 
                                 {"system_name": system_name, "error": str(e)})
            raise RuntimeError(f"Could not initialize system '{system_name}': {e}")
    
    def get_singleton_system(self, system_name: str, system_manifest: Dict[str, Any]) -> Any:
        """
        Get or create a singleton system instance.
        
        Args:
            system_name: Name of the system
            system_manifest: The system's manifest dictionary
            
        Returns:
            System instance (singleton)
        """
        if system_name in self._initialized_systems:
            return self._initialized_systems[system_name]
        
        try:
            module = importlib.import_module(system_manifest["module_path"])
            system_class = getattr(module, system_manifest["class_name"])
            instance = system_class(monitor=self.monitor)
            self._initialized_systems[system_name] = instance
            return instance
        except (ImportError, AttributeError) as e:
            self.monitor.log_event("dependency_injector", "singleton_system_failed", 
                                 {"system_name": system_name, "error": str(e)})
            raise RuntimeError(f"Could not initialize singleton system '{system_name}': {e}")

