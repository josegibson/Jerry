#!/usr/bin/env python3
"""
Jerry - Agent Instantiation Tool

Entry point for `python -m jerry` command.
"""
import sys
import os
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple

import typer
from rich.console import Console
from rich.json import JSON

# Import core business logic
from core.agent_runtime import AgentRuntime

app = typer.Typer(
    help="Jerry - AI Agent",
    add_completion=False
)

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


def _print_files_used(turn_output: Dict[str, Any]):
    """Helper to print the files used for the response."""
    used_sources = turn_output.get("used_sources")
    if not used_sources:
        return
    
    console.print("\n--- Files Used ---", style="bold yellow")
    # Filter out None or empty strings and remove duplicates
    unique_sources = sorted(list(set(s for s in used_sources if s)))
    for source in unique_sources:
        console.print(f"- {source}")
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
    console.print("\nGoodbye!")
    return True  # Signal to exit the loop


def _handle_analyze(runtime: AgentRuntime):
    console.print("\n[bold blue]Analyzing knowledge base...[/bold blue]")
    analysis = runtime.analyze_knowledge_base()
    console.print(JSON(json.dumps(analysis, indent=2)))
    return False


def _handle_reindex(runtime: AgentRuntime):
    console.print("\n[bold green]Ingesting workspace files...[/bold green]")
    stats = runtime.reindex_workspace()
    console.print(f"Ingestion complete. Indexed [green]{stats['indexed_chunks']}[/green] chunks from [green]{stats['file_count']}[/green] files.")
    return False


def _handle_history(runtime: AgentRuntime):
    console.print("\n[bold magenta]Current Session History:[/bold magenta]")
    history = runtime.get_session_history()
    console.print(history)
    return False


def _handle_config(runtime: AgentRuntime):
    console.print("\n[bold yellow]Agent Configuration:[/bold yellow]")
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

def run_interactive_cli(agent_dir: str, provider: Optional[str] = None, name: Optional[str] = None, 
                       system_prompt: Optional[str] = None):
    """
    Main interactive CLI loop for chatting with the agent.
    
    Args:
        agent_dir: Directory to run the agent in
        provider: Optional LLM provider (gemini, openai)
        name: Optional agent name
        system_prompt: Optional custom system prompt
    """
    try:
        runtime = AgentRuntime(
            agent_dir, 
            name=name, 
            provider=provider or "gemini",
            system_prompt=system_prompt
        )
        console.print("=" * 60, style="bold blue")
        console.print(f"Agent '[bold green]{runtime.config.get('name')}[/bold green]' loaded. Welcome back!")
        console.print(f"   Provider: [yellow]{runtime.config.get('provider') or 'default'}[/yellow] | Type [cyan]/help[/cyan] for commands.")
        console.print("=" * 60, style="bold blue")
    except Exception as e:
        console.print(f"[bold red]Error loading agent from '{agent_dir}': {e}[/bold red]")
        traceback.print_exc()
        return

    while True:
        try:
            user_input = console.input("\n[bold blue]You:[/bold blue] ").strip()

            if not user_input:
                continue

            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                command_name = parts[0].lower()

                command_func, _ = COMMANDS.get(command_name, (None, None))
                if command_func:
                    if command_func(runtime):
                        break  # Exit loop
                else:
                    console.print(f"[bold red]Unknown command:[/bold red] '{command_name}'. Type [cyan]/help[/cyan] for available commands.")
                continue
            
            # --- AGENT INVOCATION ---
            console.print(f"\n[bold green]{runtime.config.get('name')}:[/bold green] ", end="")
            
            turn_output = runtime.invoke(user_input)
            
            console.print(turn_output["response"])

            _print_tool_results(turn_output["tool_results"])
            _track_and_print_token_usage(runtime)
            _print_files_used(turn_output)

        except KeyboardInterrupt:
            _handle_exit(runtime)
            break
        except Exception as e:
            runtime.monitor.log_event("error", {"message": str(e), "traceback": traceback.format_exc()})
            console.print(f"[bold red]An error occurred:[/bold red] {e}")
            traceback.print_exc()


# ==============================================================================
# --- TYPER APP COMMAND ---
# ==============================================================================

@app.command()
def main(
    path: str = typer.Argument(
        ".",
        help="Path to the directory where the agent will be instantiated"
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider", "-p",
        help="LLM provider (openai or gemini). Defaults to auto-detect."
    ),
    name: Optional[str] = typer.Option(
        "Jerry",
        "--name", "-n",
        help="Name for the agent. Defaults to Jerry."
    ),
    system_prompt: Optional[str] = typer.Option(
        None,
        "--system", "-s",
        help="Custom system prompt for the agent."
    ),
    reindex: bool = typer.Option(
        False,
        "--reindex", "-r",
        help="Re-index all documents in the workspace and exit."
    )
):
    """
    Instantiate and run an AI agent in the specified directory.
    
    The agent will:
    - Index all .md and .txt files into a vector store
    - Store all state in a hidden .jerry folder
    - Provide an interactive terminal interface
    
    Examples:
        jerry .                    # Run in current directory
        jerry /path/to/project     # Run in specific directory
        jerry . --reindex          # Re-index and exit
        jerry . --provider openai  # Use specific provider
    """
    try:
        if reindex:
            console.print(f"[bold cyan]Re-indexing workspace:[/bold cyan] {path}")
            runtime = AgentRuntime(path, provider=provider, name=name, system_prompt=system_prompt)
            runtime.reindex_workspace()
            console.print("[bold green]Re-indexing complete.[/bold green]")
            return
        
        # Run interactive CLI
        run_interactive_cli(path, provider=provider, name=name, system_prompt=system_prompt)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Agent session interrupted.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    app()
