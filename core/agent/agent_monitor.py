import json
from pathlib import Path
from datetime import datetime

class AgentMonitor:
    """
    A monitoring and logging utility for an agent's lifecycle.
    """
    def __init__(self, agent_name: str, log_dir: Path):
        self.agent_name = agent_name
        self.log_dir = log_dir
        self.log_path = self.log_dir / f"{agent_name}_log.jsonl"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, data: dict):
        """
        Logs a structured event to a JSONL file.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": self.agent_name,
            "event_type": event_type,
            "data": data
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
