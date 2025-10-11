# Current Architecture of Jerry Agent System

## Overall Architecture

The Jerry Agent System is a modular, manifest-driven runtime designed for hosting AI agents. It operates on a principle of assembling agents from a set of discoverable "systems" that provide various capabilities. Each agent is self-contained within its own directory, managing its private data and declaring its required functionalities through a manifest file. A central runtime component (`AgentHost`) facilitates interaction with these agents via a command-line interface.

## Key Components

*   **AgentAssembler:**
    *   **Role:** Responsible for discovering available systems and agents, and then assembling agents by injecting their required capabilities.
    *   **Mechanism:** Scans `systems/*/manifest.json` to build a registry of available systems. When an agent is loaded, it reads `agents/<name>/manifest.json`, creates a dedicated data directory for the agent, and initializes the systems declared in `capabilities_required`.

*   **AgentHost:**
    *   **Role:** Provides an interactive command-line interface (CLI) for a single agent instance.
    *   **Mechanism:** Wraps an agent instance and runs a CLI loop, allowing users to input journal entries or execute system-defined commands.

*   **AgentSystem (Base Class for Agents):**
    *   **Role:** Serves as the foundational class for all agents, providing common functionalities and managing per-agent data storage.
    *   **Mechanism:** Automatically sets up an `EntriesDB` (SQLite-backed) at `data/entries.db` for journal-like content and provides a `save_entry(content: str)` method. Agents inheriting from `AgentSystem` gain these features by default.

*   **Systems (Pluggable Capabilities):**
    *   **Role:** Provide specific functionalities that agents can utilize (e.g., file system access, database operations, event publishing).
    *   **Mechanism:** Each system is defined by a `manifest.json` (e.g., `systems/file_system/manifest.json`) and implemented as a Python class inheriting from `BaseSystem`. They are initialized by the `AgentAssembler` and injected into the agent's context.
    *   **Examples:**
        *   **FileSystem:** Provides `writeFile(path, data)` and `readFile(path)` methods, sandboxed to the agent's directory.
        *   **DatabaseSystem:** Offers SQLite-backed key-value storage per agent, with `saveRecord` and `getRecord` methods.
        *   **EventBusSystem:** A minimal stub for publishing events, intended for future inter-agent communication.
        *   **PlannerSystem:** (Implied by `jerry` agent manifest) Provides planning capabilities with CLI commands like `:plan help`, `:plan list`, `:plan schedule`, etc.

*   **DevMonitor:**
    *   **Role:** Centralized structured event logger for the entire system.
    *   **Mechanism:** All systems log events via `DevMonitor` to `logs/system_log.jsonl`, providing a unified diagnostic stream.

*   **EntriesDB:**
    *   **Role:** A lightweight SQLite store specifically for journal-like entries, integrated into `AgentSystem`.
    *   **Mechanism:** Manages the `entries.db` file within an agent's private data directory.

## System Discovery and Assembly Flow

1.  **System Discovery:** The `AgentAssembler` scans the `systems/` directory for `manifest.json` files to identify and register all available systems.
2.  **Agent Loading:** When `load_agent(agent_name)` is called, the `AgentAssembler` reads the agent's `agents/<name>/manifest.json`.
3.  **Data Directory Creation:** A private `data/` subdirectory is created for the agent (e.g., `agents/<name>/data/`).
4.  **System Initialization:** Based on the `capabilities_required` in the agent's manifest, the `AgentAssembler` initializes the necessary systems.
    *   `database_system` receives a `db_path` pointing to `agents/<name>/data/system.db`.
    *   `file_system` receives a `root_dir` pointing to `agents/<name>`.
    *   Other systems are initialized as singletons.
5.  **Agent Context:** The agent is constructed with a context that exposes the initialized systems and its `data_dir`.

## Data Storage

*   **Agent-Private Data:** Stored under `agents/<name>/data/`. This includes:
    *   `entries.db`: Managed by `AgentSystem` for journal entries.
    *   `system.db`: Managed by `DatabaseSystem` for key-value records.
*   **Global Logs:** `logs/system_log.jsonl` contains structured events logged by `DevMonitor`.

## Manifests

*   **System Manifest (`systems/*/manifest.json`):** Defines a system's metadata (`name`, `description`, `class_name`, `module_path`).
*   **Agent Manifest (`agents/<name>/manifest.json`):** Defines an agent's identity (`agent_name`), implementation (`module_path`, `class_name`), and required capabilities (`capabilities_required`).

## Interaction and Extensibility

*   **CLI Interaction:** Users interact with agents through the `AgentHost` CLI, either by typing journal entries or using system-defined commands (e.g., `:plan`).
*   **Extending Systems:** New systems can be added by creating a directory under `systems/`, a `manifest.json`, and implementing the system's Python class inheriting from `BaseSystem`.
*   **Extending Agents:** New agents can be created by setting up a directory under `agents/`, a `manifest.json`, and implementing the agent's Python class (optionally inheriting from `AgentSystem`).
