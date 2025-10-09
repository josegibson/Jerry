from pathlib import Path
from typing import Any

from systems.agent.entries_db import EntriesDB


class AgentSystem:
	"""Base class for all agents with default per-agent storage behavior."""
	def __init__(self, context: Any, data_dir: Path | None = None):
		self.context = context
		# Determine agent data directory from context or argument
		if data_dir is None:
			data_dir_attr = getattr(context, "data_dir", None)
			if data_dir_attr is None:
				raise ValueError("AgentSystem requires 'data_dir' in context or as an argument")
			self.data_dir = Path(data_dir_attr)
		else:
			self.data_dir = Path(data_dir)
		self.data_dir.mkdir(parents=True, exist_ok=True)

		# Built-in entries database for journal-like content
		self.entries_db = EntriesDB(self.data_dir / "entries.db")

	def save_entry(self, content: str) -> str:
		"""Saves an entry to the agent's entries.db and returns the entry id."""
		return self.entries_db.add_entry(content)

	def on_start(self):
		"""Hook called by the runtime host when the agent starts."""
		pass

	def shutdown(self):
		"""Shutdown hook to close resources."""
		try:
			self.entries_db.close()
		except Exception:
			pass

	def on_stop(self):
		"""Alias for shutdown; called by the runtime host on stop."""
		self.shutdown()


