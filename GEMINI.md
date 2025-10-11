# Jerry Agent System - Project Overview

Jerry is a modular, manifest-driven agent runtime for Windows. It assembles AI agents from discoverable "systems" such as file system, database, and event bus. Each agent operates within its own directory, possessing private data and a manifest that outlines its required capabilities. A lightweight runtime hosts the agent and offers a straightforward command-line interface for interaction.

## Building and Running

### Setup
1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running
*   **System Shell (multi-agent entrypoint):**
    *   To run the interactive system shell that loads all agents discovered under `agents/*/manifest.json`:
        ```bash
        python __main__.py
        ```
    *   To start a shell for a specific agent only (e.g., `jerry`):
        ```bash
        python __main__.py --agent jerry
        ```
*   **Single Agent Runner:**
    *   To run a specific agent by directory name using its manifest and the agent host CLI (e.g., `jerry`):
        ```bash
        python systems/runtime/run_agent.py jerry
        ```

## Development Conventions

*   **Modular Architecture:** The system is built around "systems" (pluggable capabilities) and "agents" (AI entities).
*   **Manifest-Driven Configuration:** Both systems and agents are configured via `manifest.json` files, defining their names, descriptions, class names, module paths, and required capabilities.
*   **Logging:** All system events are logged via `DevMonitor` to `logs/system_log.jsonl` for diagnostics.
*   **Base Classes:**
    *   `BaseSystem`: Base class for all pluggable systems.
    *   `AgentSystem`: Base class for agents, providing default per-agent storage behavior and an `EntriesDB`.
*   **CLI Commands:** Systems can expose CLI commands by implementing a `get_cli_commands()` method.
*   **Data Storage:** Agent-private data is stored under `agents/<agent_name>/data/`.

## Key Components

*   **AgentAssembler:** Discovers available systems and assembles agents based on their manifests.
*   **AgentHost:** Provides an interactive CLI environment for a single agent instance.
*   **DevMonitor:** Centralized structured event logger.
*   **Systems:** Pluggable capabilities like `FileSystem`, `DatabaseSystem`, `EventBusSystem`, and `PlannerSystem`.
*   **EntriesDB:** A lightweight SQLite store for journal-like entries, integrated into `AgentSystem`.

## Directory Structure

```
D:\Jerry\
  __main__.py                     # System shell entrypoint (multi-agent)
  systems\                       # Contains definitions for various pluggable systems
    agent\
      agent_system.py             # Base agent with entries store
      entries_db.py               # Lightweight SQLite for journal entries
    assembler\agent_assembler.py  # Discovers systems, builds agents
    database\                    # Database system
    dev_monitor\dev_monitor.py    # Structured logging
    event_bus\                   # Event bus system
    file_system\                 # File system system
    planner\                     # Planner system
    runtime\                     # Agent runtime components
      agent_host.py               # CLI host for a single agent
      run_agent.py                # Run a single agent by name
  agents\                        # Contains definitions and data for individual agents
    jerry\                       # Example agent
      manifest.json               # Agent manifest
      data\                      # Private data (entries.db, system.db)
  logs\system_log.jsonl           # DevMonitor log
  requirements.txt                # Project dependencies
  README.md                       # Project documentation
```