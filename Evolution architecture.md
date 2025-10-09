Project Plan: Milestone 1 - The Bedrock
Objective: To refactor the existing codebase into a modular, system-based architecture and create a single, portable, Stage 1 agent ("Jerry") that serves as the foundation for the entire ecosystem.

Architectural Principles for this Milestone
Solitary Agent First: We will build a single, useful agent before introducing multi-agent complexity. The initial input stream is a direct function call, not an event bus.

Portability is Paramount: An agent's entire existence (identity, logic, and data) is contained within its directory.

Security by Design: All systems inherit from a SystemBase class, which enforces a mandatory, manifest-driven permission check for every action.

Dynamic Discovery: The AgentRuntime discovers and integrates systems and agents dynamically by reading their manifests, not through hardcoded registries.

Task 1: Establish Foundational Systems
The first step is to formalize the core logic from the existing core module into standalone, secure "System" modules within a new /systems/ directory.

1.1. Create the SystemBase Class:

File: /systems/base.py.

Action: Create an abstract base class. Its primary responsibility is to be initialized with an agent_manifest and provide an internal _check_permission() method. This is the security contract for all future systems.

1.2. Create the FileSystem System:

File: /systems/file_system.py.

Action: Create a FileSystem class that inherits from SystemBase. Refactor the logic from your existing core/tools/file_tools.py into this class. Its methods (write, read, etc.) will be scoped to the agent's private directory and must call _check_permission() before executing.

1.3. Create the DatabaseSystem System:

File: /systems/database_system.py.

Action: Create a DatabaseSystem class that inherits from SystemBase. Refactor the non-vector, data storage logic (e.g., using SQLite for portability) into this class. Its methods (save_record, get_record, etc.) will operate on a database file within the agent's directory and must call _check_permission().

1.4. Create the SystemManifest Standard:

Action: For each system, create a system_manifest.json file in its directory. This allows the AgentRuntime to dynamically discover available systems and the capabilities they provide.

Task 2: Define the Agent Standard
2.1. Refactor the AgentRuntime:

Action: Refactor the existing core/AgentRuntime.py. Its new, focused responsibility is to act as the "agent assembler":

Scan /systems/ to build a registry of available capabilities.

Load an agent by reading its /agents/<agent_name>/manifest.json.

For each capability requested in the manifest, initialize the corresponding system class with the agent's manifest and inject it as a tool.

2.2. Formalize the AgentManifest:

Action: Finalize the structure of the agent_manifest_template.json. This file is the agent's identity and contract with the ecosystem.

Task 3: Build the First Agent ("Jerry")
3.1. Create the Jerry Directory Structure:

Create /agents/jerry/.

Inside, create /agents/jerry/data/ for its private data storage.

Place the manifest.json file inside /agents/jerry/.

3.2. Configure Jerry's Manifest:

Fill out the manifest. Crucially, the "capabilities_required" list will only contain two items for this milestone: "file_system" and "database_system".

3.3. Implement Jerry's Logic:

Create /agents/jerry/logic.py. This will contain a JerryAgent class with a simple function like add_journal_entry(text), which uses the database_system tool provided by the runtime.

Task 4: A Look Ahead (Future Milestones)
This foundational work enables the natural evolution of the system.

Milestone 2: The Society: We will create the EventBusSystem. This capability will be granted to Jerry when it needs to create its first child agent, providing a private communication channel for its "family."

Milestone 3: The Timekeeper: We will create the PlannerSystem as a parallel tree. This will be granted to agents when the Cortex detects the need for time-based tasks and scheduling.