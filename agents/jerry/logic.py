from systems.agent.agent_system import AgentSystem

class JerryAgent(AgentSystem):
    """
    Jerry is the primary agent for interacting with the user.
    """
    def __init__(self, context):
        super().__init__(context)
        # uses context.entries_db provided by the runtime

    def addJournalEntry(self, text: str):
        """
        Adds a journal entry to the database.
        """
        entry_id = self.save_entry(text)
        print("[Jerry] Saved new journal entry.")
