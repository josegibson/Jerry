from systems.base_system import BaseSystem
from systems.dev_monitor.dev_monitor import DevMonitor

class EventBusSystem(BaseSystem):
    def __init__(self, monitor: DevMonitor):
        super().__init__(monitor)
    
    def publish(self, agent_id: str, event_name: str, event_data: dict):
        # In a real implementation, this would publish to a message queue
        self.log_event("published", {"agent_id": agent_id, "event_name": event_name, "event_data": event_data})
        print(f"[EventBus] Event '{event_name}' published by '{agent_id}' with data: {event_data}")
