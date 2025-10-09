import sys
import dotenv
import typer
import inspect
from rich.console import Console

from core.agent.agent_runtime import AgentRuntime
from agents.jerry import JerryAgent
from agents.conrad import ConradAgent

# --- Configuration ---
dotenv.load_dotenv()
console = Console()
AGENT_CLASSES = {
    "jerry": JerryAgent,
    "conrad": ConradAgent,
}

# --- Helper Functions ---

def _parse_command(user_input: str):
    """Parses 'method(args)' or 'agent.method(args)'."""
    try:
        # Split agent/method from args
        command_part, args_part = user_input.split('(', 1)
        
        # Extract args string
        if not args_part.endswith(')'):
            return None, None, None
        arg_str = args_part[:-1]

        # Parse agent and method
        if '.' in command_part:
            agent_name, method_name = command_part.split('.', 1)
        else:
            agent_name, method_name = None, command_part

        # Rudimentary arg parsing (handles a single string)
        args = []
        if arg_str:
            if arg_str.startswith('"') and arg_str.endswith('"'):
                args.append(arg_str[1:-1])
            # TODO: Add other types like numbers if needed
        
        return agent_name, method_name, args
    except ValueError:
        return None, None, None

# --- Shell Implementations ---

def _run_agent_shell(runtime: AgentRuntime, agent_name: str):
    """Runs an interactive shell for a single, pre-loaded agent."""
    agent = runtime.get_agent(agent_name)
    if not agent:
        console.print(f"[bold red]Agent '{agent_name}' could not be loaded.[/bold red]")
        return

    console.print(f"✅ Agent Shell for [bold green]'{agent_name}'[/bold green] is ready. Type 'help' or 'exit'.")
    
    while True:
        try:
            user_input = console.input(f"\n[bold green]{agent_name}>[/bold green] ").strip()
            if not user_input:
                continue
            if user_input.lower() == 'exit':
                break
            
            _, method_name, args = _parse_command(user_input)

            if user_input.lower() == 'help':
                console.print(f"\n[bold underline]Available commands for {agent_name}:[/bold underline]")
                for name, method in inspect.getmembers(agent, predicate=inspect.ismethod):
                    if not name.startswith('_'):
                        doc = inspect.getdoc(method) or "No description."
                        console.print(f"  [cyan]{name}[/cyan]: {doc.strip().split('\n')[0]}")
                continue

            method = getattr(agent, method_name, None)
            if not callable(method):
                console.print(f"[bold red]Unknown command: '{method_name}'. Type 'help' for available commands.[/bold red]")
                continue

            method(*args)

        except Exception as e:
            console.print(f"[bold red]❌ Error executing command: {e}[/bold red]")

def _run_system_shell(runtime: AgentRuntime):
    """Runs the main system shell for multi-agent interaction."""
    console.print("✅ System Shell is ready. Type 'exit' to quit.")
    console.print("   Example: jerry.addJournalEntry(\"Today I worked on the POC.\")")

    while True:
        try:
            user_input = console.input("\n[bold green]>[/bold green] ").strip()
            if not user_input:
                continue
            if user_input.lower() == 'exit':
                break

            agent_name, method_name, args = _parse_command(user_input)

            if not agent_name:
                console.print("[bold red]Invalid command format. Use 'agent_name.method_name(args)'.[/bold red]")
                continue

            agent = runtime.get_agent(agent_name)
            if not agent:
                console.print(f"[bold red]Agent '{agent_name}' not found.[/bold red]")
                continue

            method = getattr(agent, method_name, None)
            if not callable(method):
                console.print(f"[bold red]Method '{method_name}' not found on agent '{agent_name}'.[/bold red]")
                continue
            
            method(*args)

        except Exception as e:
            console.print(f"[bold red]❌ Error executing command: {e}[/bold red]")


# --- Main Application ---

def main(agent: str = typer.Option(None, "--agent", help="Start a shell for a specific agent.")):
    """
    Main entry point for the Jerry System Shell.
    """
    console.print("=" * 60, style="bold blue")
    console.print("🚀 System Shell Initializing...")
    
    project_root = "D:\\Jerry" 
    
    try:
        runtime = AgentRuntime(project_root)
        
        if agent:
            # Agent-specific shell mode
            agent_class = AGENT_CLASSES.get(agent.lower())
            if not agent_class:
                console.print(f"[bold red]Unknown agent: '{agent}'. Available: {list(AGENT_CLASSES.keys())}[/bold red]")
                return
            runtime.load_agent(agent.lower(), agent_class)
            _run_agent_shell(runtime, agent.lower())
        else:
            # System-wide shell mode
            console.print("Loading all agents...")
            for name, agent_class in AGENT_CLASSES.items():
                runtime.load_agent(name, agent_class)
            _run_system_shell(runtime)

        runtime.shutdown()
        console.print("=" * 60, style="bold blue")
        console.print("👋 Goodbye!")

    except Exception as e:
        console.print(f"❌ [bold red]Critical error during runtime: {e}[/bold red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    typer.run(main)