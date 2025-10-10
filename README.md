## Jerry Agent System

### Overview
Jerry is a modular, manifest-driven agent runtime for Windows that assembles agents from discoverable "systems" (file system, database, event bus, etc.). Each agent lives entirely in its own directory with private data and a manifest that declares required capabilities. A thin runtime hosts the agent and provides a simple CLI for interaction.

### Key Concepts
- **AgentAssembler**: Discovers systems via `systems/*/manifest.json`, loads an agent from `agents/<name>/manifest.json`, initializes required systems, and injects them into the agent's context.
- **AgentHost**: Wraps a single agent instance and runs an interactive CLI.
- **AgentSystem (base for agents)**: Provides per-agent data directory management and a built-in `entries.db` via `EntriesDB`. Exposes `save_entry(content: str)` and lifecycle hooks.
- **Systems**: Pluggable capabilities (e.g., `file_system`, `database_system`, `event_bus`) with their own manifests. All systems inherit from `BaseSystem` and log via `DevMonitor`.
- **DevMonitor**: Centralized structured event logger writing to `logs/system_log.jsonl`.

### Directory Layout
```
d:\Jerry\
  __main__.py                     # System shell entrypoint (multi-agent)
  systems\
    base_system.py                # Base class for systems
    dev_monitor\dev_monitor.py    # Structured logging
    assembler\agent_assembler.py  # Discovers systems, builds agents
    runtime\agent_host.py         # CLI host for a single agent
    runtime\run_agent.py          # Run a single agent by name
    file_system\
      manifest.json               # { name, class_name, module_path }
      file_system.py              # FileSystem system
    database\
      manifest.json               # DatabaseSystem manifest
      database_system.py          # SQLite-backed KV per agent
    event_bus\
      manifest.json               # EventBusSystem manifest
      event_bus_system.py         # Simple publish stub
    agent\
      agent_system.py             # Base agent with entries store
      entries_db.py               # Lightweight SQLite for journal entries
  agents\
    jerry\
      manifest.json               # Declares capabilities and agent class
      data\                       # Private data (entries.db, system.db)
  logs\system_log.jsonl           # DevMonitor log
  requirements.txt
  Evolution architecture.md        # Design notes
```

### Requirements
- Windows 10/11
- Python 3.13 (recommended; a `jerry_env` venv is included in this repo structure)

### Setup (Windows)
1) (Optional) Use the provided venv:
```bat
cd D:\Jerry
jerry_env\Scripts\activate
```
2) Install dependencies:
```bat
pip install -r requirements.txt
```

### Running
- **System Shell (multi-agent entrypoint):**
  - Run the interactive system shell that loads all agents discovered under `agents/*/manifest.json`.
  ```bat
  python __main__.py
  ```
  - To start a shell for a specific agent only:
  ```bat
  python __main__.py --agent jerry
  ```

- **Single Agent Runner:**
  - Run a specific agent by directory name using its manifest and the agent host CLI.
  ```bat
  python systems\runtime\run_agent.py jerry
  ```

### CLI Usage
- In the **system shell**, commands must be of the form `agent.method("args")`.
  - Example to save a journal entry to agent `jerry`:
  ```
  jerry.save_entry("Today I worked on the POC.")
  ```
  - Type `exit` to quit.

- In the **single agent host** (when running `--agent jerry` or `run_agent.py jerry`), you can either type your journal text per line, or use system-defined commands. Type `exit` to quit.
  - Journal: type any text and press Enter → saved via `save_entry`.
  - System-defined commands: each capability system may expose CLI commands by implementing `get_cli_commands()` returning entries like `{ prefix, help, handler }`. The host aggregates these and dispatches inputs to the appropriate system.
  - Example (PlannerSystem), shown only when the agent requires `planner_system`:
    - `:plan help`
    - `:plan list`
    - `:plan schedule "<desc>" <YYYY-MM-DD> <HH:MM> [priority=<int>] [recurrence=daily|weekly|monthly] [interval=<int>]`
    - `:plan complete <task_id>`
    - `:plan cancel <task_id>`

### How It Works
#### System Discovery and Assembly
- `AgentAssembler` scans `systems/*/manifest.json` to build a registry:
```startLine:endLine:systems/assembler/agent_assembler.py
def _discover_systems(self):
	"""Scans the systems directory for manifests and populates the registry."""
	print("Discovering systems...")
	for system_dir in self.systems_dir.iterdir():
		manifest_path = system_dir / "manifest.json"
		if manifest_path.is_file():
			...
```
- When `load_agent(agent_name)` is called, it reads `agents/<name>/manifest.json`, creates `data/`, and initializes required systems. Special cases:
  - `database_system`: receives `db_path` pointing to `agents/<name>/data/system.db`
  - `file_system`: receives `root_dir` pointing to `agents/<name>`
  - Other systems are initialized as singletons.

#### Agent Context and Base
- The agent is constructed with a context exposing raw systems and a `data_dir` hint. Agents that inherit `AgentSystem` automatically get an `EntriesDB` at `data/entries.db` and `save_entry(content: str)`.
```startLine:endLine:systems/agent/agent_system.py
class AgentSystem:
	"""Base class for all agents with default per-agent storage behavior."""
	...
	# Built-in entries database for journal-like content
	self.entries_db = EntriesDB(self.data_dir / "entries.db")

	def save_entry(self, content: str) -> str:
		"""Saves an entry to the agent's entries.db and returns the entry id."""
		return self.entries_db.add_entry(content)
```

#### Hosting and CLI
- `AgentHost` provides a CLI loop for a single agent (journal + system-defined commands):
```startLine:endLine:systems/runtime/agent_host.py
def run_cli(self, prompt_label: str = "agent"):
	"""Run a simple blocking CLI loop supporting journal input and :plan commands."""
	self.console.print("✅ CLI ready. Type your journal entry, or use :plan. Type 'exit' to quit.")
	while True:
		...
```

### Manifests
#### System Manifest (`systems/*/manifest.json`)
Example:
```json
{
  "name": "file_system",
  "description": "Provides tools for interacting with the agent's private file system.",
  "class_name": "FileSystem",
  "module_path": "systems.file_system.file_system"
}
```
Notes:
- CLI help for planner commands is capability-gated and will only appear if `planner_system` is declared and initialized for the agent.
- Systems can add commands by implementing `get_cli_commands()` on the system object in `agent.context`. Each item should include:
  - `prefix: str` (e.g., `:plan`)
  - `help: str` (short help suffix that will be shown)
  - `handler: Callable[[str, Console, Any], None]` that receives the raw `user_input`, the rich `Console`, and the `agent` instance.

#### Agent Manifest (`agents/<name>/manifest.json`)
The agent manifest defines the agent identity, the class to instantiate, and the capabilities it requires. Current schema (as used by `agents/jerry/manifest.json`):
```json
{
  "agent_name": "jerry",
  "description": "The foundational agent.",
  "module_path": "systems.agent.agent_system",
  "class_name": "AgentSystem",
  "capabilities_required": [
    "file_system",
    "planner_system"
  ]
}
```
Notes:
- `agent_name`: Identifier for the agent directory and logs.
- `module_path` and `class_name`: The import path and class name to instantiate. You can point these to your own agent class (e.g., `agents.jerry.logic.JerryAgent`) or use the base `AgentSystem` if its defaults are sufficient.
- `capabilities_required`: List of system `name` values from their manifests (e.g., `file_system`, `database_system`, `event_bus`). Only declare what your agent needs. The example above uses just `file_system`, matching the current codebase.

### Built-in Systems and APIs
#### FileSystem (`systems.file_system.file_system.FileSystem`)
- Sandbox root: the agent directory (e.g., `agents/jerry/`).
- Methods:
  - `writeFile(path: str, data: str) -> str`
  - `readFile(path: str) -> str`
```startLine:endLine:systems/file_system/file_system.py
def writeFile(self, path: str, data: str) -> str:
	...
	return f"Successfully wrote to file '{path}'."

def readFile(self, path: str) -> str:
	...
	return safe_path.read_text(encoding='utf-8')
```

#### DatabaseSystem (`systems.database.database_system.DatabaseSystem`)
- SQLite-backed key-value per agent (scoped by `agent_id + collection`).
- Methods:
  - `saveRecord(agent_id: str, collection: str, record: dict)`
  - `getRecord(agent_id: str, collection: str, record_id: str) -> dict | None`
```startLine:endLine:systems/database/database_system.py
def saveRecord(self, agent_id: str, collection: str, record: dict):
	...
	self.conn.commit()

def getRecord(self, agent_id: str, collection: str, record_id: str) -> dict | None:
	...
	return json.loads(row[0])
```

#### EventBusSystem (`systems.event_bus.event_bus_system.EventBusSystem`)
- Minimal publish stub for future inter-agent comms.
- Method:
  - `publish(event_name: str, event_data: dict)`

#### EntriesDB (`systems.agent.entries_db.EntriesDB`)
- Lightweight SQLite store for journal-like entries used by `AgentSystem`.
```startLine:endLine:systems/agent/entries_db.py
def add_entry(self, content: str) -> str:
	entry_id = content[:20]
	...
	return entry_id
```

### Logging and Diagnostics
- All systems log via `DevMonitor` to `logs/system_log.jsonl`.
```startLine:endLine:systems/dev_monitor/dev_monitor.py
def log_event(self, source: str, event_type: str, data: dict):
	...
	with open(self.log_path, "a", encoding="utf-8") as f:
		f.write(json.dumps(log_entry) + "\n")
```
- View recent events by opening `logs/system_log.jsonl`. Systems and runtime also print key messages to the console.

### Extending the Project
#### Create a New System
1) Create a directory under `systems/<your_system>/` with a `manifest.json`:
```json
{
  "name": "my_system",
  "description": "What it does",
  "class_name": "MySystem",
  "module_path": "systems.my_system.my_system"
}
```
2) Implement `MySystem(BaseSystem)` in `systems/my_system/my_system.py` and accept a `monitor: DevMonitor` in `__init__`.
3) On next run, it will be auto-discovered and available for agents via `capabilities_required`.

#### Create a New Agent
1) Create `agents/<name>/` with `data/` subdirectory.
2) Add `manifest.json` with `class_name`, `module_path`, and `capabilities_required`.
3) Implement the agent class. For convenience, inherit from `AgentSystem` to get `entries.db` and `save_entry`.

### Data Storage
- Agent-private data lives under `agents/<name>/data/`:
  - `entries.db` (created by `AgentSystem`)
  - `system.db` (created by `DatabaseSystem` when required)
- Global logs are under `logs/system_log.jsonl`.

### Troubleshooting (Windows)
- If imports fail, ensure you are running from the project root (`D:\Jerry`) so relative module paths resolve.
- If you see `Manifest ... not found`, confirm `agents/<name>/manifest.json` exists and is valid JSON.
- For permission errors when reading/writing files, check that the paths are inside the agent workspace and not attempting directory traversal.

### Notes
- The architecture and milestones are further described in `Evolution architecture.md`.
- This project is designed to evolve into a multi-agent system with an event bus and planner in future milestones.

