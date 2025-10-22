from typing import Dict, Optional
from rich.console import Console

from core.agent.agent_system import AgentSystem
from core.loader.agent_loader import AgentLoader
from .cli_interface import CLIInterface


class AgentRuntime:
    """
    Main runtime coordinator for agent interaction.
    Manages agent loading and provides interface coordination.
    """
    
    def __init__(self, root_dir: str, console: Optional[Console] = None):
        """
        Initialize the agent runtime.
        
        Args:
            root_dir: Root directory of the project
            console: Rich console for output (optional)
        """
        self.console = console or Console()
        self.loader = AgentLoader(root_dir)
        self.loaded_agents: Dict[str, AgentSystem] = {}
    
    def load_agent(self, agent_name: str) -> AgentSystem:
        """
        Load an agent by name.
        
        Args:
            agent_name: Name of the agent to load
            
        Returns:
            Loaded agent instance
        """
        agent = self.loader.load_agent(agent_name)
        self.loaded_agents[agent_name] = agent
        return agent
    
    def get_agent(self, agent_name: str) -> Optional[AgentSystem]:
        """
        Get a loaded agent by name.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent instance or None if not found
        """
        return self.loaded_agents.get(agent_name)
    
    def create_cli_interface(self, agent_name: str) -> CLIInterface:
        """
        Create a CLI interface for a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            CLI interface instance
            
        Raises:
            ValueError: If agent not found
        """
        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found.")
        
        return CLIInterface(agent, self.console)
    
    def run_agent_cli(self, agent_name: str, prompt_label: Optional[str] = None):
        """
        Run CLI for a specific agent.
        
        Args:
            agent_name: Name of the agent
            prompt_label: Custom prompt label (defaults to agent name)
        """
        cli = self.create_cli_interface(agent_name)
        cli.start()
        try:
            cli.run_cli(prompt_label or agent_name)
        finally:
            cli.stop()
    
    def list_loaded_agents(self) -> list[str]:
        """Get a list of loaded agent names."""
        return list(self.loaded_agents.keys())
    
    def shutdown(self):
        """Shutdown the runtime and all loaded agents."""
        for agent in self.loaded_agents.values():
            if hasattr(agent, "shutdown"):
                agent.shutdown()
        self.loader.shutdown()
        self.console.print("Runtime shutdown complete.")

