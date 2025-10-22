import json
from pathlib import Path
from typing import Dict, Any

from core.monitoring.dev_monitor import DevMonitor


class SystemRegistry:
    """
    Discovers and manages available systems by scanning core/system/*/manifest.json
    """
    
    def __init__(self, systems_dir: Path, monitor: DevMonitor):
        """
        Initialize the system registry.
        
        Args:
            systems_dir: Path to the systems directory (core/system/)
            monitor: DevMonitor instance for logging
        """
        self.systems_dir = Path(systems_dir)
        self.monitor = monitor
        self.systems_registry: Dict[str, Dict[str, Any]] = {}
        self._discover_systems()
    
    def _discover_systems(self):
        """Scans the systems directory for manifests and populates the registry."""
        print("Discovering systems...")
        for system_dir in self.systems_dir.iterdir():
            if not system_dir.is_dir():
                continue
                
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
                    self.monitor.log_event("runtime_error", "system_discovery_failed", 
                                         {"path": str(manifest_path), "error": str(e)})
    
    def get_system_manifest(self, system_name: str) -> Dict[str, Any]:
        """
        Get the manifest for a specific system.
        
        Args:
            system_name: Name of the system
            
        Returns:
            The system manifest dictionary
            
        Raises:
            ValueError: If system not found
        """
        if system_name not in self.systems_registry:
            raise ValueError(f"System '{system_name}' not found in registry.")
        return self.systems_registry[system_name]
    
    def list_available_systems(self) -> list[str]:
        """Get a list of all available system names."""
        return list(self.systems_registry.keys())
    
    def is_system_available(self, system_name: str) -> bool:
        """Check if a system is available."""
        return system_name in self.systems_registry

