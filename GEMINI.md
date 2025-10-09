# Gemini Project Context: Jerry AI Framework

## Project Overview

This project, "Jerry," is a Python-based framework for developing and running modular, multi-agent AI systems. The long-term vision, as outlined in `Evolution architecture.md`, is to create a highly modular, portable, and secure ecosystem where agents are self-contained entities that dynamically discover and use "Systems" (like file systems, databases, etc.) based on manifest files.

The current implementation is a significant step towards this vision. It features a central `AgentRuntime` that manages the lifecycle of agents and the systems they use. While it doesn't yet use manifest-driven discovery, it establishes the core concepts of separate systems and an agent runtime.

**Key Technologies:**
*   **Language:** Python 3.10+
*   **CLI:** Typer, Rich
*   **Core Logic:** LangChain, LangGraph for building agent reasoning graphs.
*   **LLM Providers:** Google Gemini, OpenAI (configurable via `.env`).
*   **RAG/Vector Store:** ChromaDB.
*   **Dependencies:** Managed via `requirements.txt`.

## Building and Running

### 1. Installation
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory to store API keys.
```dotenv
# .env
GOOGLE_API_KEY="your-google-api-key"
OPENAI_API_KEY="your-openai-api-key"
```

### 3. Running the System
The application is run as a Python module from the project root. It has two main modes:

**A) System-Wide Shell:**
This mode loads all available agents and allows you to interact with them in a single shell.
```bash
python -m core
```
Once running, you can call methods on loaded agents, for example: `jerry.addJournalEntry("This is a test.")`

**B) Single-Agent Shell:**
This mode starts an interactive shell for one specific agent.
```bash
python -m core --agent jerry
```
Once running, you can execute commands directly, for example: `addJournalEntry("This is a test.")`

## Development Conventions

### Target Architecture (`Evolution architecture.md`)
*   **Systems:** The goal is for core capabilities (FileSystem, Database) to be standalone modules in a `/systems` directory.
*   **Agents:** Agents should be fully portable directories (`/agents/<agent_name>/`) containing their logic, data, and a `manifest.json`.
*   **Dynamic Discovery:** An `AgentRuntime` should dynamically load agents and the systems they require by reading their manifest files, enforcing a permission model.

### Current Implementation
*   **`AgentRuntime` (`core/agent/agent_runtime.py`):** This is the central orchestrator. It currently manually initializes and holds instances of all systems (`DatabaseSystem`, `PlannerSystem`, etc.).
*   **`BaseSystem` (`core/base_system.py`):** A base class that all systems inherit from, providing common functionality like monitoring.
*   **`SystemProxy` (`core/agent/agent_runtime.py`):** A clever proxy class that sits between an agent and a system. It automatically injects the `agent_id` into every system call, ensuring that systems operate within the correct agent's scope (e.g., saving data to the right agent's database table).
*   **Agent Definition (`agents/`):** Agents (e.g., `JerryAgent`) inherit from the base `Agent` class. They are initialized with an `AgentContext` object, which gives them access to the system proxies (e.g., `self.context.db.saveRecord(...)`).
*   **Entrypoint (`core/__main__.py`):** This file, using Typer, serves as the main entry point and CLI. It initializes the `AgentRuntime` and starts one of the interactive shells.
