from core.agent.agent_runtime import Agent

class JerryAgent(Agent):
    """
    Jerry is the primary agent for interacting with the user.
    """
    def addJournalEntry(self, text: str):
        """
        Adds a journal entry to the database and publishes an event.
        """
        print("[Jerry] Received journal entry. NOTE: Database and EventBus are currently disconnected.")
        # entry = {"id": text[:20], "content": text} # simple id for now
        # self.runtime.systems["db"].saveRecord("entries", entry)
        # self.runtime.systems["event_bus"].publish("NewJournalEntry", {"content": text})
        print("[Jerry] Pretended to save new journal entry.")
