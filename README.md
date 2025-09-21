# Jerry: A Modular, Multi-Agent AI Framework

Jerry is a Python-based framework for building and managing multiple, independent AI agents. The core architectural principle is that each agent is a self-contained unit, living within its own directory. This design provides a clean separation of concerns, allowing each agent to have its own knowledge base, tools, and configuration.

Powered by LangChain and LangGraph, Jerry provides a robust foundation for creating sophisticated, tool-using AI assistants that can reason, retrieve information, and interact with a local file system.

## Design Philosophy

The framework is designed with a strict separation between the core agent logic and the user interface. 

- **`AgentRuntime`**: This class, located in `core/agent_runtime.py`, encapsulates all of the agent's core functionality, including state management, graph invocation, and lifecycle events like startup and shutdown. It provides a clean, high-level API (`invoke`, `shutdown`, etc.) and has no knowledge of the user interface.
- **`__main__.py`**: This acts as a pure command-line interface (CLI). Its sole responsibility is to accept user input, call the appropriate methods on the `AgentRuntime`, and present the results to the user. 

This decoupled architecture makes the system highly maintainable and allows for the easy creation of new interfaces (e.g., a web UI, a Discord bot) by simply building a new view layer on top of the existing `AgentRuntime`.

## Core Features

* **Directory-Scoped Agents**: Each agent is an independent entity defined by a directory, containing its own configuration and a dedicated vector store.
* **Ephemeral Sessions**: To ensure clean, repeatable interactions, each time you run an agent is a new session. The agent's conversational memory is cleared upon shutdown.
* **Session Archiving**: On shutdown, a complete, human-readable Markdown transcript of the conversation is automatically saved to the `workspace/sessions/` directory, named with a timestamp.
* **Retrieval-Augmented Generation (RAG)**: Agents automatically index Markdown files within their `workspace/` directory into a ChromaDB vector store. This allows them to answer questions based on a private knowledge base, while intelligently ignoring archived session files.
* **Extensible Tool System**: A `ToolProvider` architecture makes it easy to grant agents new capabilities, such as file system access (`file_tools`), knowledge retrieval (`knowledge_tools`), and web search.
* **Pluggable Backends**: Easily switch between LLM providers (e.g., Gemini, OpenAI) and embedding models (e.g., OpenAI, HuggingFace) via environment variables.
* **Structured Logging**: A built-in `AgentMonitor` tracks token usage and logs all major events (graph execution, tool calls, errors) to a structured JSON log file in the agent's `logs/` directory.

## Getting Started

### 1. Prerequisites

* Python 3.10+
* Pip for package management

### 2. Installation

Clone the repository and install the required dependencies:

```bash
git clone <your-repo-url>
cd <your-repo-directory>
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the project's root directory. This file will store your API keys and other configuration settings.

```dotenv
# .env

# --- Required for LLM Providers (choose at least one) ---
# For Google Gemini
GOOGLE_API_KEY="your-google-api-key"

# For OpenAI
OPENAI_API_KEY="your-openai-api-key"

# --- Optional for Embedding Models ---
# If OPENAI_API_KEY is set, it defaults to "text-embedding-3-small".
# To use a local model instead, ensure you have sentence-transformers installed.
# SENTENCE_TRANSFORMERS_MODEL="all-MiniLM-L6-v2"
```

### 4. Creating and Running Your First Agent

The framework is designed to be run as a module from the root of the project. To create a new agent, simply provide a path to a directory that doesn't exist yet. The framework will automatically scaffold it for you.

```bash
# This will create a new agent named "my-first-agent" in a new directory
python -m core ./my-first-agent
```

After the initial setup, you can start chatting with your agent. To run an existing agent, just point to its directory:

```bash
# Run the agent located in the ./my-first-agent directory
python -m core ./my-first-agent
```

-----

## Usage

Once an agent is running, you can interact with it through the command-line interface.

### Chatting

Simply type your message and press Enter.

### Special Commands

The CLI now supports a range of commands prefixed with `/` for agent management and interaction. The terminal experience is enhanced with rich text formatting and syntax highlighting provided by the `rich` library.

  * `/quit`, `/exit`, `/q`: Shuts down the agent, archives the conversation to a Markdown file, and clears the session state for the next run.
  * `/analyze`: Performs an analysis of the agent's knowledge base, showing file counts, token estimates, and vector store statistics.
  * `/ingest`: Rescans the workspace and ingests new or updated documents into the agent's knowledge base.
  * `/history`: Displays the full conversation history for the current session.
  * `/config`: Shows the agent's current configuration.
  * `/clear`: Clears the console screen.
  * `/help`: Displays a list of all available commands and their descriptions.

### Adding Knowledge

To give your agent new knowledge, simply create or copy `.md` (Markdown) files into its `workspace/` subdirectory (e.g., `./my-first-agent/workspace/`). The agent will automatically detect and index these files the next time it starts.

-----

## How It Works

### The Agent Directory Structure

When you create an agent, the following structure is generated:

```
my-first-agent/
├── .chroma/              # Sandboxed ChromaDB vector store for long-term knowledge
├── .agent_state.json     # Ephemeral conversation history (cleared on shutdown)
├── logs/                 # Agent-specific operational logs
│   └── agent_my-first-agent.log
├── workspace/            # Your private knowledge base (add .md files here)
│   └── sessions/         # Contains archived Markdown transcripts of past conversations
└── agent.json            # The agent's configuration file
```

### Agent Configuration (`agent.json`)

This file defines the agent's core properties. You can edit it to change the agent's behavior.

```json
{
  "name": "my-first-agent",
  "provider": "gemini", // or "openai"
  "system_prompt": "You are a helpful assistant specialized in software development.",
  "retrieve_top_k": 5,
  "tools": [
    "file_tools",
    "knowledge_tools",
    "web_search_tools"
  ]
}
```