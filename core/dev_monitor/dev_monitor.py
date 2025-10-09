import json
from pathlib import Path
from datetime import datetime

class DevMonitor:
    """
    A centralized monitoring and logging utility for the entire system.
    """
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_path = self.log_dir / "system_log.jsonl"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_event(self, source: str, event_type: str, data: dict):
        """
        Logs a structured event to a JSONL file.

        Args:
            source: The source of the event (e.g., 'agent_jerry', 'system_database').
            event_type: The type of event (e.g., 'initialized', 'record_saved').
            data: A dictionary of event-specific data.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "event_type": event_type,
            "data": data
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")