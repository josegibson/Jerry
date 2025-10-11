import argparse
from pathlib import Path
from rich.console import Console

from core.assembler.agent_assembler import AgentAssembler
from core.runtime.agent_host import AgentHost


def main():
	parser = argparse.ArgumentParser(description="Run an agent by name using its manifest")
	parser.add_argument("agent", help="Agent name (directory name under agents/) to run")
	args = parser.parse_args()

	console = Console()
	project_root = str(Path(__file__).resolve().parents[2])

	assembler = AgentAssembler(project_root)
	agent_name = args.agent.lower()

	# Load the agent via its manifest
	agent = assembler.load_agent(agent_name)

	# Host runtime and CLI
	host = AgentHost(agent, console)
	host.start()
	try:
		host.run_cli(agent_name)
	finally:
		host.stop()
		assembler.shutdown()
		console.print("👋 Goodbye!")


if __name__ == "__main__":
	main()


