from typing import Any, Optional
from rich.console import Console

from core.agent.agent_system import AgentSystem


class CommandDispatcher:
    """
    Dispatches commands to appropriate handlers.
    Handles both system commands and agent method calls.
    """
    
    def __init__(self, console: Console):
        """
        Initialize the command dispatcher.
        
        Args:
            console: Rich console for output
        """
        self.console = console
    
    def dispatch_agent_command(self, agent: AgentSystem, method_name: str, args: list) -> Any:
        """
        Dispatch a command to an agent method.
        
        Args:
            agent: The agent instance
            method_name: Name of the method to call
            args: Arguments to pass to the method
            
        Returns:
            Result of the method call
            
        Raises:
            AttributeError: If method doesn't exist on agent
        """
        method = getattr(agent, method_name, None)
        if not callable(method):
            raise AttributeError(f"Method '{method_name}' not found on agent.")
        
        return method(*args)
    
    def dispatch_system_command(self, user_input: str, agent: AgentSystem) -> bool:
        """
        Dispatch a system command (like :plan).
        
        Args:
            user_input: Raw user input
            agent: The agent instance
            
        Returns:
            True if command was handled, False otherwise
        """
        context = getattr(agent, "context", object())
        
        # Look for system commands
        for attr in dir(context):
            if attr.startswith("_"):
                continue
            sys_obj = getattr(context, attr, None)
            if sys_obj and hasattr(sys_obj, "get_cli_commands") and callable(getattr(sys_obj, "get_cli_commands")):
                try:
                    commands = sys_obj.get_cli_commands() or []
                    for cmd in commands:
                        if isinstance(cmd, dict) and "prefix" in cmd and "handler" in cmd:
                            prefix = cmd.get("prefix")
                            handler = cmd.get("handler")
                            if isinstance(prefix, str) and user_input.startswith(prefix) and callable(handler):
                                try:
                                    handler(user_input, self.console, agent)
                                    return True
                                except Exception as e:
                                    self.console.print(f"[bold red]❌ Command error: {e}[/bold red]")
                                    return True
                except Exception:
                    pass
        
        return False
    
    def execute_command(self, agent: AgentSystem, method_name: str, args: list) -> Any:
        """
        Execute a command on an agent.
        
        Args:
            agent: The agent instance
            method_name: Name of the method to call
            args: Arguments to pass to the method
            
        Returns:
            Result of the method call
        """
        try:
            result = self.dispatch_agent_command(agent, method_name, args)
            if result is not None:
                self.console.print(result)
            return result
        except AttributeError as e:
            self.console.print(f"[bold red]Method '{method_name}' not found on agent.[/bold red]")
            raise
        except Exception as e:
            self.console.print(f"[bold red]❌ Error executing command: {e}[/bold red]")
            raise

