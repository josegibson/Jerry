from .dev_monitor.dev_monitor import DevMonitor

class BaseSystem:
    """
    The blueprint class for all systems.
    """
    def __init__(self, monitor: DevMonitor):
        self.monitor = monitor
        self.system_name = self.__class__.__name__.lower()
        self.monitor.log_event(f"system_{self.system_name}", "initialized", {})

    def log_event(self, event_type: str, data: dict):
        """A helper method to log events through the system's monitor."""
        self.monitor.log_event(f"system_{self.system_name}", event_type, data)