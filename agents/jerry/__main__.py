import sys
from pathlib import Path
from rich.console import Console

from systems.assembler.agent_assembler import AgentAssembler
from systems.runtime.agent_host import AgentHost


def main():
	console = Console()

	# Resolve project root as the repository root (two levels up from this file)
	project_root = str(Path(__file__).resolve().parents[2])

	assembler = AgentAssembler(project_root)
	agent_name = "jerry"
	agent = assembler.load_agent(agent_name)

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


