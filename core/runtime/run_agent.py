import argparse
from pathlib import Path
from rich.console import Console

from core.runtime.agent_runtime import AgentRuntime


def main():
	parser = argparse.ArgumentParser(description="Run an agent by name using its manifest")
	parser.add_argument("agent", help="Agent name (directory name under agents/) to run")
	args = parser.parse_args()

	console = Console()
	project_root = str(Path(__file__).resolve().parents[2])

	runtime = AgentRuntime(project_root, console)
	agent_name = args.agent.lower()

	try:
		# Load and run the agent
		runtime.load_agent(agent_name)
		runtime.run_agent_cli(agent_name)
	finally:
		runtime.shutdown()
		console.print("👋 Goodbye!")


if __name__ == "__main__":
	main()


