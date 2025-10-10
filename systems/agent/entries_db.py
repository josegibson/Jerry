import sqlite3
import uuid
from pathlib import Path
from datetime import datetime


class EntriesDB:
	"""Lightweight per-agent SQLite store for journal entries."""
	def __init__(self, db_path: Path):
		self.db_path = str(db_path)
		self.conn = sqlite3.connect(self.db_path)
		self._ensure_schema()

	def _ensure_schema(self):
		cursor = self.conn.cursor()
		# Create with created_at when initializing fresh
		cursor.execute(
			"""
			CREATE TABLE IF NOT EXISTS entries (
				id TEXT PRIMARY KEY,
				content TEXT NOT NULL,
				created_at TEXT NOT NULL
			)
			"""
		)
		# Migrate existing tables lacking created_at
		cursor.execute("PRAGMA table_info(entries)")
		columns = [row[1] for row in cursor.fetchall()]
		if "created_at" not in columns:
			try:
				cursor.execute("ALTER TABLE entries ADD COLUMN created_at TEXT")
			except sqlite3.OperationalError:
				# Column may already exist due to race or previous migration
				pass
		self.conn.commit()

	def add_entry(self, content: str) -> str:
		entry_id = str(uuid.uuid4())
		cursor = self.conn.cursor()
		created_at = datetime.utcnow().isoformat()
		# Use column list to be robust across migrations
		cursor.execute("INSERT OR REPLACE INTO entries (id, content, created_at) VALUES (?, ?, ?)", (entry_id, content, created_at))
		self.conn.commit()
		return entry_id

	def close(self):
		try:
			self.conn.close()
		except Exception:
			pass


