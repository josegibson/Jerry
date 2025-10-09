from typing import Optional
from rich.console import Console


class AgentHost:
	"""Thin runtime wrapper for a single agent instance."""
	def __init__(self, agent, console: Optional[Console] = None):
		self.agent = agent
		self.console = console or Console()

	def start(self):
		"""Start the agent runtime (lifecycle hook)."""
		if hasattr(self.agent, "on_start"):
			self.agent.on_start()

	def stop(self):
		"""Stop the agent runtime (lifecycle hook)."""
		if hasattr(self.agent, "on_stop"):
			self.agent.on_stop()

	def run_cli(self, prompt_label: str = "agent"):
		"""Run a simple blocking CLI loop that forwards input to addJournalEntry."""
		self.console.print("✅ CLI ready. Type your journal entry, or 'exit' to quit.")
		while True:
			try:
				user_input = self.console.input(f"\n[bold green]{prompt_label}>[/bold green] ").strip()
				if not user_input:
					continue
				if user_input.lower() == "exit":
					break

				# Delegate to the agent's method
				self.agent.save_entry(user_input)
				self.console.print("[bold green]Journal entry added.[/bold green]")
			except KeyboardInterrupt:
				break
			except Exception as e:
				self.console.print(f"[bold red]❌ Error: {e}[/bold red]")


