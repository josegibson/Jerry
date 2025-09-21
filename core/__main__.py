import sys
import dotenv
import json
import os
from typing import Dict, Any, List, Optional, Callable, Tuple
import traceback

import typer
from rich.console import Console
from rich.json import JSON
from rich.text import Text

from .agent_runtime import AgentRuntime

# Load environment variables from a .env file at the project root
dotenv.load_dotenv()

console = Console()

# ==============================================================================
# --- CLI HELPER FUNCTIONS ---
# ==============================================================================

def _print_tool_results(tool_results: Optional[List[Dict[str, Any]]]):
    """Helper to print tool results if they exist."""
    if not tool_results:
        return
    
    console.print("\n--- Tool Results ---", style="bold yellow")
    for res in tool_results:
        console.print(f"Tool: [green]{res.get('tool_name')}[/green]")
        console.print(f"Input: [cyan]{res.get('tool_input')}[/cyan]")
        console.print(f"Output: [magenta]{res.get('output')}[/magenta]")
        if res.get('error'):
            console.print(f"Error: [red]{res.get('error')}[/red]")
    console.print("--------------------", style="bold yellow")


def _track_and_print_token_usage(runtime: AgentRuntime):
    session_metrics = runtime.monitor.get_token_metrics()
    session_total = session_metrics.get("session_total", {"input": 0, "output": 0, "total": 0})
    console.print(
        f"\n[bold blue]Token Usage:[/bold blue] "
        f"Prompt=[bold]{session_total.get('input', 0)}[/bold] | "
        f"Completion=[bold]{session_total.get('output', 0)}[/bold] | "
        f"Total=[bold]{session_total.get('total', 0)}[/bold]"
    )

# ==============================================================================
# --- CLI COMMAND HANDLERS ---
# ==============================================================================

def _handle_exit(runtime: AgentRuntime):
    console.print("\n[bold yellow]Shutting down and archiving session...[/bold yellow]")
    shutdown_summary = runtime.shutdown()
    console.print(shutdown_summary["message"])
    console.print("\n👋 Goodbye!")
    return True # Signal to exit the loop

def _handle_analyze(runtime: AgentRuntime):
    console.print("\n[bold blue]🔬 Analyzing knowledge base...[/bold blue]")
    analysis = runtime.analyze_knowledge_base()
    console.print(JSON(json.dumps(analysis, indent=2)))
    return False

def _handle_reindex(runtime: AgentRuntime):
    console.print("\n[bold green]🔄 Ingesting workspace files...[/bold green]")
    stats = runtime.reindex_workspace()
    console.print(f"Ingestion complete. Indexed [green]{stats['indexed_chunks']}[/green] chunks from [green]{stats['file_count']}[/green] files.")
    return False

def _handle_history(runtime: AgentRuntime):
    console.print("\n[bold magenta]📜 Current Session History:[/bold magenta]")
    history = runtime.get_session_history()
    console.print(history)
    return False

def _handle_config(runtime: AgentRuntime):
    console.print("\n[bold yellow]⚙️ Agent Configuration:[/bold yellow]")
    config = runtime.get_config()
    console.print(JSON(json.dumps(config, indent=2)))
    return False

def _handle_clear(runtime: AgentRuntime):
    os.system('cls' if os.name == 'nt' else 'clear')
    console.print("[bold green]Console cleared.[/bold green]")
    return False

def _handle_help(runtime: AgentRuntime):
    console.print("\n[bold underline]Available Commands:[/bold underline]")
    for cmd, (_, desc) in COMMANDS.items():
        console.print(f"  [cyan]{cmd}[/cyan]: {desc}")
    return False

COMMANDS: Dict[str, Tuple[Callable[[AgentRuntime], bool], str]] = {
    "/quit": (_handle_exit, "Exit the agent session."),
    "/exit": (_handle_exit, "Exit the agent session."),
    "/q": (_handle_exit, "Exit the agent session."),
    "/analyze": (_handle_analyze, "Analyze the current knowledge base."),
    "/reindex": (_handle_reindex, "Rescan the workspace and ingest new/updated documents."),
    "/history": (_handle_history, "Show the messages from the current session."),
    "/config": (_handle_config, "Display the agent's configuration file."),
    "/clear": (_handle_clear, "Clear the console screen."),
    "/help": (_handle_help, "Show this help message."),
}

# ==============================================================================
# --- MAIN CLI LOOP ---
# ==============================================================================

def run_cli(agent_dir: str):
    """
    Main CLI entry point for interacting with an agent.
    """
    try:
        runtime = AgentRuntime(agent_dir)
        console.print("=" * 60, style="bold blue")
        console.print(f"🤖 Agent '[bold green]{runtime.config.get('name')}[/bold green]' loaded. Welcome back!")
        console.print(f"   Provider: [yellow]{runtime.config.get('provider') or 'default'}[/yellow] | Type [cyan]/help[/cyan] for commands.")
        console.print("=" * 60, style="bold blue")
    except Exception as e:
        console.print(f"❌ [bold red]Error loading agent from '{agent_dir}': {e}[/bold red]")
        traceback.print_exc()
        return

    while True:
        try:
            user_input = console.input("\n[bold blue]📝 You:[/bold blue] ").strip()

            if not user_input:
                continue

            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                command_name = parts[0].lower()
                command_args = parts[1:]

                command_func, _ = COMMANDS.get(command_name, (None, None))
                if command_func:
                    # For now, commands don't take args directly from CLI in this simple dispatch
                    # If commands needed args, we'd parse command_args and pass them.
                    if command_func(runtime):
                        break # Exit loop
                else:
                    console.print(f"[bold red]Unknown command:[/bold red] '{command_name}'. Type [cyan]/help[/cyan] for available commands.")
                continue
            
            # --- AGENT INVOCATION ---
            console.print(f"\n[bold green]🤖 {runtime.config.get('name')}:[/bold green] ", end="")
            
            turn_output = runtime.invoke(user_input)
            
            console.print(turn_output["response"])

            _print_tool_results(turn_output["tool_results"])
            _track_and_print_token_usage(runtime)

        except KeyboardInterrupt:
            _handle_exit(runtime)
            break
        except Exception as e:
            runtime.monitor.log_event("error", {"message": str(e), "traceback": traceback.format_exc()})
            console.print(f"[bold red]❌ An error occurred:[/bold red] {e}")
            traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cli(sys.argv[1])
    else:
        print("Usage: python -m core <path_to_agent_directory>")
        print("Example: python -m core ./my-first-agent")

