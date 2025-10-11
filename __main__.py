
import sys
import dotenv
import typer
import inspect
from pathlib import Path
from rich.console import Console

from core.assembler.agent_assembler import AgentAssembler
from core.runtime.agent_host import AgentHost

# --- Configuration ---
dotenv.load_dotenv()
console = Console()

# --- Helper Functions (largely unchanged) ---

def _parse_command(user_input: str):
    """Parses 'method(args)' or 'agent.method(args)'."""
    try:
        command_part, args_part = user_input.split('(', 1)
        if not args_part.endswith(')'):
            return None, None, None
        arg_str = args_part[:-1]

        if '.' in command_part:
            agent_name, method_name = command_part.split('.', 1)
        else:
            agent_name, method_name = None, command_part

        args = []
        if arg_str:
            # This simple parsing handles a single string argument.
            # It can be improved to handle multiple args, numbers, etc.
            if arg_str.startswith('"') and arg_str.endswith('"'):
                args.append(arg_str[1:-1])
            else:
                args.append(arg_str) # Treat as a single value
        
        return agent_name, method_name, args
    except ValueError:
        return None, None, None

# --- Shell Implementations (largely unchanged) ---

def _run_agent_shell(runtime, agent_name: str):
    """Runs an interactive shell for a single, pre-loaded agent using AgentHost."""
    agent = runtime.get_agent(agent_name)
    if not agent:
        console.print(f"[bold red]Agent '{agent_name}' could not be loaded.[/bold red]")
        return

    host = AgentHost(agent, console)
    host.start()
    try:
        host.run_cli(agent_name)
    finally:
        host.stop()

def _run_system_shell(runtime):
    """Runs the main system shell for multi-agent interaction."""
    console.print("✅ System Shell is ready. Type 'exit' to quit.")
    console.print("   Example: jerry.save_entry(\"Today I worked on the POC.\")")
    # Display loaded agents and their capabilities if available
    try:
        loaded = getattr(runtime, "agents", {})
        if loaded:
            rows = []
            for name, agent in loaded.items():
                caps = sorted(list(getattr(agent, "capabilities", set())))
                rows.append(f"   - {name}: [{', '.join(caps)}]")
            console.print("Loaded agents and capabilities:")
            for row in rows:
                console.print(row)
    except Exception:
        # Non-fatal if runtime changes structure
        pass

    while True:
        try:
            user_input = console.input("\n[bold green]>[/bold green] ").strip()
            if not user_input: continue
            if user_input.lower() == 'exit': break
            # monitor commands removed

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
            
            result = method(*args)
            if result is not None:
                console.print(result)

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
    agents_dir = Path(project_root) / "agents"
    
    try:
        runtime = AgentAssembler(project_root)
        
        # Discover available agents by looking for manifest.json files
        available_agents = [d.name for d in agents_dir.iterdir() if d.is_dir() and (d / 'manifest.json').is_file()]
        
        if agent:
            # Agent-specific shell mode
            agent_name = agent.lower()
            if agent_name not in available_agents:
                console.print(f"[bold red]Unknown agent: '{agent_name}'. Available: {available_agents}[/bold red]")
                return
            runtime.load_agent(agent_name)
            _run_agent_shell(runtime, agent_name)
        else:
            # System-wide shell mode
            console.print("Loading all agents...")
            if not available_agents:
                console.print("[yellow]No agents found. Create an agent with a manifest.json to begin.[/yellow]")
            for agent_name in available_agents:
                try:
                    runtime.load_agent(agent_name)
                except Exception as e:
                    console.print(f"[bold red]Failed to load agent '{agent_name}': {e}[/bold red]")
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
