from typing import Optional
from rich.console import Console

from core.agent.agent_system import AgentSystem
from .command_parser import CommandParser
from .command_dispatcher import CommandDispatcher


class CLIInterface:
    """
    Handles CLI interaction for a single agent.
    Manages command parsing, dispatching, and user interaction.
    """
    
    def __init__(self, agent: AgentSystem, console: Optional[Console] = None):
        """
        Initialize the CLI interface.
        
        Args:
            agent: The agent instance to interface with
            console: Rich console for output (optional)
        """
        self.agent = agent
        self.console = console or Console()
        self.command_parser = CommandParser()
        self.command_dispatcher = CommandDispatcher(self.console)
    
    def start(self):
        """Start the agent runtime (lifecycle hook)."""
        if hasattr(self.agent, "on_start"):
            self.agent.on_start()
    
    def stop(self):
        """Stop the agent runtime (lifecycle hook)."""
        if hasattr(self.agent, "on_stop"):
            self.agent.on_stop()
    
    def run_cli(self, prompt_label: str = "agent"):
        """
        Run the main CLI loop.
        
        Args:
            prompt_label: Label for the CLI prompt
        """
        self.console.print("✅ CLI ready. Type your journal entry. Type 'exit' to quit.")
        
        # Collect and display system commands
        self._display_system_commands()
        
        while True:
            try:
                user_input = self.console.input(f"\n[bold green]{prompt_label}>[/bold green] ").strip()
                if not user_input:
                    continue
                if user_input.lower() == "exit":
                    break
                
                # Try system command first
                if self.command_dispatcher.dispatch_system_command(user_input, self.agent):
                    continue
                
                # Try agent method call
                agent_name, method_name, args = self.command_parser.parse_command(user_input)
                if method_name:
                    try:
                        self.command_dispatcher.execute_command(self.agent, method_name, args)
                        continue
                    except Exception:
                        # If method call fails, treat as journal entry
                        pass
                
                # Otherwise, treat as journal entry
                self.agent.save_entry(user_input)
                self.console.print("[bold green]Journal entry added.[/bold green]")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print(f"[bold red]❌ Error: {e}[/bold red]")
    
    def _display_system_commands(self):
        """Display available system commands."""
        commands = []
        context = getattr(self.agent, "context", object())
        
        for attr in dir(context):
            if attr.startswith("_"):
                continue
            sys_obj = getattr(context, attr, None)
            if sys_obj and hasattr(sys_obj, "get_cli_commands") and callable(getattr(sys_obj, "get_cli_commands")):
                try:
                    for cmd in sys_obj.get_cli_commands() or []:
                        if isinstance(cmd, dict) and "prefix" in cmd and "handler" in cmd:
                            commands.append(cmd)
                except Exception:
                    pass
        
        # Display help for each command provider
        for cmd in commands:
            help_line = cmd.get("help")
            if help_line:
                self.console.print(f"   {cmd['prefix']} {help_line}")

