
import sys
import dotenv
import typer
from pathlib import Path
from rich.console import Console

from core.runtime.agent_runtime import AgentRuntime
from core.runtime.command_parser import CommandParser
from core.runtime.command_dispatcher import CommandDispatcher

# --- Configuration ---
dotenv.load_dotenv()
console = Console()

# --- Helper Functions ---

def _run_system_shell(runtime: AgentRuntime):
	"""Runs the main system shell for multi-agent interaction."""
	console.print("✅ System Shell is ready. Type 'exit' to quit.")
	console.print("   Example: jerry.save_entry(\"Today I worked on the POC.\")")
	
	# Display loaded agents and their capabilities
	loaded_agents = runtime.list_loaded_agents()
	if loaded_agents:
		console.print("Loaded agents and capabilities:")
		for agent_name in loaded_agents:
			agent = runtime.get_agent(agent_name)
			if agent:
				caps = sorted(list(getattr(agent, "capabilities", set())))
				console.print(f"   - {agent_name}: [{', '.join(caps)}]")

	command_parser = CommandParser()
	command_dispatcher = CommandDispatcher(console)

	while True:
		try:
			user_input = console.input("\n[bold green]>[/bold green] ").strip()
			if not user_input: 
				continue
			if user_input.lower() == 'exit': 
				break

			agent_name, method_name, args = command_parser.parse_command(user_input)

			if not agent_name:
				console.print("[bold red]Invalid command format. Use 'agent_name.method_name(args)'.[/bold red]")
				continue

			agent = runtime.get_agent(agent_name)
			if not agent:
				console.print(f"[bold red]Agent '{agent_name}' not found.[/bold red]")
				continue

			# Try system command first
			if command_dispatcher.dispatch_system_command(user_input, agent):
				continue

			# Try agent method call
			try:
				command_dispatcher.execute_command(agent, method_name, args)
			except Exception as e:
				console.print(f"[bold red]❌ Error executing command: {e}[/bold red]")

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
		runtime = AgentRuntime(project_root, console)
		
		# Discover available agents by looking for manifest.json files
		available_agents = [d.name for d in agents_dir.iterdir() if d.is_dir() and (d / 'manifest.json').is_file()]
		
		if agent:
			# Agent-specific shell mode
			agent_name = agent.lower()
			if agent_name not in available_agents:
				console.print(f"[bold red]Unknown agent: '{agent_name}'. Available: {available_agents}[/bold red]")
				return
			runtime.load_agent(agent_name)
			runtime.run_agent_cli(agent_name)
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
