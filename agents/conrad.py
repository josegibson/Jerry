from core.agent.agent_runtime import Agent, AgentContext

class ConradAgent(Agent):
    """
    Conrad is an agent that listens for events and provides analysis.
    """
    def __init__(self, context: AgentContext):
        super().__init__(context)
        print("[Conrad] NOTE: EventBus is currently disconnected.")
        # self.context.event_bus.subscribe("NewJournalEntry", self.onNewJournalEntry)

    def onNewJournalEntry(self, payload: dict):
        """
        Callback for when a new journal entry is created.
        """
        print("[Conrad] Received new journal entry. Scanning for career keywords...")
        # In a real scenario, Conrad would perform some analysis here.
        content = payload.get("content", "").lower()
        if "resume" in content or "python project" in content:
            print("[Conrad] Career-related keywords found in journal entry.")

    def showStatus(self):
        print("[Conrad] Status: Listening for events.")
