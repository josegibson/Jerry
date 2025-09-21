# Jerry: A Modular, Multi-Agent AI Framework

Jerry is a Python-based framework for building and managing multiple, independent AI agents. The core architectural principle is that each agent is a self-contained unit, living within its own directory. This design provides a clean separation of concerns, allowing each agent to have its own persistent memory, knowledge base, tools, and configuration.

Powered by LangChain and LangGraph, Jerry provides a robust foundation for creating sophisticated, tool-using AI assistants that can reason, retrieve information, and interact with a local file system.




## Core Features

* **Directory-Scoped Agents**: Each agent is an independent entity defined by a directory, containing its own configuration, long-term memory, and a dedicated vector store.
* **Retrieval-Augmented Generation (RAG)**: Agents automatically index Markdown files within their `workspace` directory into a ChromaDB vector store, allowing them to answer questions based on a private knowledge base.
* **Persistent Conversations**: Chat history is automatically saved and loaded, allowing you to resume conversations with an agent at any time.
* **Extensible Tool System**: A `ToolProvider` architecture makes it easy to grant agents new capabilities, such as file system access (`file_tools`), knowledge retrieval (`knowledge_tools`), and web search.
* **Pluggable Backends**: Easily switch between LLM providers (e.g., Gemini, OpenAI) and embedding models (e.g., OpenAI, HuggingFace) via environment variables.
* **Monitoring & Logging**: A built-in `AgentMonitor` tracks token usage, logs important events, and provides tools to analyze an agent's knowledge base.


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
````

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

### 4\. Creating and Running Your First Agent

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

```
📝 You: Can you list all files in a directory?
```

### Special Commands

  * `quit`, `exit`, `q`: Shuts down the agent and saves the final session metrics.
  * `!analyze`: Performs an analysis of the agent's knowledge base, showing file counts, token estimates, and vector store statistics.

### Adding Knowledge

To give your agent new knowledge, simply create or copy `.md` (Markdown) files into its `workspace` subdirectory (e.g., `./my-first-agent/workspace/`). The agent will automatically detect and index these files the next time it starts, making their content available for retrieval.

-----

## How It Works

### The Agent Directory Structure

When you create an agent, the following structure is generated:

```
my-first-agent/
├── .chroma/              # Sandboxed ChromaDB vector store
├── .agent_state.json     # Serialized conversation history
├── logs/                 # Agent-specific operational logs
│   └── agent_my-first-agent.log
├── workspace/            # Your private knowledge base (add .md files here)
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

This modular design allows you to create and maintain a diverse ecosystem of specialized AI agents, each tailored to a specific task or domain.

```