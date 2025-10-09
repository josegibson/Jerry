import sqlite3
from pathlib import Path


class EntriesDB:
	"""Lightweight per-agent SQLite store for journal entries."""
	def __init__(self, db_path: Path):
		self.db_path = str(db_path)
		self.conn = sqlite3.connect(self.db_path)
		self._ensure_schema()

	def _ensure_schema(self):
		cursor = self.conn.cursor()
		cursor.execute(
			"""
			CREATE TABLE IF NOT EXISTS entries (
				id TEXT PRIMARY KEY,
				content TEXT NOT NULL
			)
			"""
		)
		self.conn.commit()

	def add_entry(self, content: str) -> str:
		entry_id = content[:20]
		cursor = self.conn.cursor()
		cursor.execute("INSERT OR REPLACE INTO entries (id, content) VALUES (?, ?)", (entry_id, content))
		self.conn.commit()
		return entry_id

	def close(self):
		try:
			self.conn.close()
		except Exception:
			pass


