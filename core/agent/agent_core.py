from pathlib import Path
from typing import Any, Optional

from .agent_system import AgentSystem


class AgentCore:
    """
    Minimal agent initialization and management.
    Handles the bare necessities for creating and managing agent instances.
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialize the agent core with a data directory.
        
        Args:
            data_dir: Path to the agent's data directory
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def create_agent(self, context: Any) -> AgentSystem:
        """
        Create a new agent instance with the given context.
        
        Args:
            context: The agent context containing systems and capabilities
            
        Returns:
            A new AgentSystem instance
        """
        return AgentSystem(context, self.data_dir)
    
    def get_data_dir(self) -> Path:
        """Get the agent's data directory path."""
        return self.data_dir

