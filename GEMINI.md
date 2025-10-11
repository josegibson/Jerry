# Project: Jerry Agent System

## Project Overview

The Jerry Agent System is a modular, manifest-driven agent runtime designed for Windows environments. It allows for the assembly of AI agents from a collection of discoverable "systems" such as file system access, database interactions, and an event bus. Each agent operates within its own dedicated directory, managing private data and declaring its required capabilities through a manifest file. A lightweight runtime hosts these agents and provides a command-line interface for interaction. The system emphasizes modularity, allowing for easy extension with new systems and agents.

**Key Components:**
*   **AgentAssembler:** Discovers and loads systems and agents based on their manifest files.
*   **AgentHost:** Provides an interactive CLI for a single agent instance.
*   **AgentSystem:** A base class for agents, offering per-agent data directory management and an integrated `entries.db` for journal-like content.
*   **Systems:** Pluggable capabilities (e.g., `file_system`, `database_system`, `event_bus`, `planner`) that inherit from `BaseSystem`.
*   **DevMonitor:** Centralized structured event logger for system diagnostics.

**Main Technologies:**
*   Python 3.13
*   SQLite (for `entries.db` and `system.db`)

## Building and Running

### Requirements
*   Windows 10/11
*   Python 3.13 (a `jerry_env` virtual environment is provided)

### Setup
1.  **Activate Virtual Environment (Optional but Recommended):**
    ```bash
    cd D:\Jerry
    jerry_env\Scripts\activate
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the System

*   **System Shell (Multi-Agent Entrypoint):**
    To run the interactive system shell that loads all agents discovered under `agents/*/manifest.json`:
    ```bash
    python __main__.py
    ```
*   **Specific Agent Shell:**
    To start a shell for a specific agent (e.g., "jerry"):
    ```bash
    python __main__.py --agent jerry
    ```
*   **Single Agent Runner:**
    To run a specific agent by its directory name using its manifest and the agent host CLI:
    ```bash
    python systems\runtime\run_agent.py jerry
    ```

### CLI Usage

*   **In the System Shell:**
    Commands are in the format `agent.method("args")`.
    Example to save a journal entry to agent `jerry`:
    ```
    jerry.save_entry("Today I worked on the POC.")
    ```
    Type `exit` to quit.

*   **In the Single Agent Host:**
    (When running `--agent jerry` or `run_agent.py jerry`)
    You can type journal text directly (saved via `save_entry`) or use system-defined commands. Type `exit` to quit.
    System-defined commands are prefixed with a colon (e.g., `:plan`). These are capability-gated and appear only if the agent requires the corresponding system.

## Development Conventions

*   **Modularity:** The system is built around modular "systems" and "agents," each defined by a `manifest.json` file.
*   **System Manifests:** Located under `systems/*/manifest.json`, they define the system's name, description, class name, and module path.
*   **Agent Manifests:** Located under `agents/<name>/manifest.json`, they define the agent's name, description, class name, module path, and `capabilities_required`.
*   **Logging:** All systems log events via `DevMonitor` to `logs/system_log.jsonl`.
*   **Data Storage:** Agent-private data is stored under `agents/<name>/data/` (e.g., `entries.db`, `system.db`).
*   **Extensibility:** New systems and agents can be added by creating appropriate directories and manifest files, and implementing the corresponding Python classes.