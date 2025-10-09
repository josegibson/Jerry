from core.agent.agent_runtime import Agent, AgentContext

class JerryAgent(Agent):
    """
    Jerry is the primary agent for interacting with the user.
    """
    def __init__(self, context: AgentContext):
        super().__init__(context)

    def addJournalEntry(self, text: str):
        """
        Adds a journal entry to the database and publishes an event.
        """
        entry = {"id": text[:20], "content": text} # simple id for now
        self.context.db.saveRecord("entries", entry)
        # self.context.event_bus.publish("NewJournalEntry", {"content": text})
        print("[Jerry] Saved new journal entry.")
