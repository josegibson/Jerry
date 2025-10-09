import sqlite3
import json
from pathlib import Path

from systems.base_system import BaseSystem
from systems.dev_monitor.dev_monitor import DevMonitor

class DatabaseSystem(BaseSystem):
    """
    A simple key-value database system, scoped by agent ID.
    """
    def __init__(self, monitor: DevMonitor, db_path: str):
        super().__init__(monitor)
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.log_event("initialized", {"db_path": self.db_path})

    def _get_scoped_collection(self, agent_id: str, collection: str) -> str:
        """Creates a unique, safe collection name for an agent."""
        return f"{agent_id}_{collection}"

    def saveRecord(self, agent_id: str, collection: str, record: dict):
        """Saves a record for a specific agent."""
        scoped_collection = self._get_scoped_collection(agent_id, collection)
        record_id = record.get("id")
        if not record_id:
            raise ValueError("Record must have an 'id' field.")

        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {scoped_collection} (
                id TEXT PRIMARY KEY,
                data TEXT
            )
        """)
        
        self.cursor.execute(f"INSERT OR REPLACE INTO {scoped_collection} (id, data) VALUES (?, ?)", (record_id, json.dumps(record)))
        self.conn.commit()
        self.log_event("record_saved", {"agent_id": agent_id, "collection": collection, "record_id": record_id})

    def getRecord(self, agent_id: str, collection: str, record_id: str) -> dict | None:
        """Retrieves a record for a specific agent."""
        scoped_collection = self._get_scoped_collection(agent_id, collection)
        
        # Check if table exists first to prevent errors
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (scoped_collection,))
        if self.cursor.fetchone() is None:
            return None

        self.cursor.execute(f"SELECT data FROM {scoped_collection} WHERE id = ?", (record_id,))
        row = self.cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            self.log_event("connection_closed", {})
