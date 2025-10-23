# Jerry: AI Agent Instantiation Tool

Jerry is a Python-based tool that enables you to instantiate an AI agent from **any directory** in your file system. The agent automatically indexes (vectorizes) all readable documents (`.md` and `.txt` files) into a vector store, augmenting its responses with semantically relevant chunks from your documents.

**Key Feature**: All agent state data is confined to a hidden `.jerry` folder, keeping your workspace clean and uncluttered.

## 📚 Documentation Quick Links

| Guide | Description |
|-------|-------------|
| **[INSTALL.md](INSTALL.md)** | 🚀 Quick installation guide |
| **[QUICKSTART.md](docs/QUICKSTART.md)** | ⚡ Get running in 5 minutes |
| **[USER_GUIDE.md](docs/USER_GUIDE.md)** | 📖 Complete user journey with examples |
| **[CONTRIBUTING.md](docs/CONTRIBUTING.md)** | 💻 Developer setup and contribution guide |

## Design Philosophy

Jerry operates with a simple principle: **any directory can become an intelligent workspace**. Just navigate to your project directory and run the tool. The agent will:

- Index all your markdown and text files
- Store all its state in a hidden `.jerry` folder
- Provide context-aware responses based on your documents
- Operate entirely from the terminal

The framework is designed with a strict separation between the core agent logic and the user interface:

- **`AgentRuntime`**: Encapsulates all agent functionality (state management, graph invocation, lifecycle events)
- **CLI Interface**: Provides a clean terminal interface for user interaction
- **Modular Architecture**: Easy to extend with new tools and capabilities

## Core Features

* **Directory-Agnostic**: Run Jerry from any directory in your file system - no special setup required
* **Hidden State Management**: All agent data (vector store, logs, configs) stored in a clean `.jerry` folder
* **Automatic Document Indexing**: Scans and indexes all `.md` and `.txt` files for RAG-enhanced responses
* **Ephemeral Sessions**: Each session is fresh; conversation history is cleared on shutdown
* **Session Archiving**: Conversations automatically saved to `.jerry/sessions/` as Markdown transcripts
* **Retrieval-Augmented Generation (RAG)**: Semantic search over your documents with ChromaDB
* **Extensible Tool System**: File operations, knowledge retrieval, and web search capabilities
* **Multiple LLM Providers**: Switch between Gemini, OpenAI, and other providers via flags or environment variables
* **Structured Logging**: Comprehensive logging with token tracking in `.jerry/logs/`

## Getting Started

### 1. Prerequisites

* Python 3.9+ (Python 3.10+ recommended)
* pip (Python package installer)

### 2. Installation

```bash
git clone https://github.com/josegibson/Jerry.git
cd Jerry
pip install -e .
```

That's it! The `jerry` command is now available globally.

> **Note**: See [INSTALL.md](INSTALL.md) for detailed installation options and troubleshooting.

### 3. Configuration

Create a `.env` file in the project's root directory with your API keys:

```bash
# Required: At least one LLM provider
GOOGLE_API_KEY="your-google-api-key"
OPENAI_API_KEY="your-openai-api-key"

# Optional: Web search
TAVILY_API_KEY="your-tavily-api-key"
```

### 4. Basic Usage

**Once installed, you can run Jerry from anywhere:**

```bash
# Navigate to any directory with documents
cd /path/to/your/project

# Start the agent (uses current directory)
jerry

# Or specify a directory
jerry /path/to/another/project

# Choose a specific provider
jerry --provider openai

# Custom agent name
jerry --name "MyProjectBot"
```

**What happens:**
1. Jerry creates a hidden `.jerry` folder in your directory
2. All `.md` and `.txt` files are automatically indexed
3. You can chat with the agent, which has context from your documents
4. Everything is stored in `.jerry/` - your workspace stays clean!

-----

## Usage

Once Jerry is running, you can interact with it through the command-line interface.

### Chatting

Simply type your message and press Enter. The agent has access to all the content in your `.md` and `.txt` files and will provide contextually relevant responses.

### Special Commands

The CLI supports commands prefixed with `/` for agent management:

  * `/quit`, `/exit`, `/q`: Shut down the agent, archive the conversation, and clear session state
  * `/analyze`: Analyze the knowledge base (file counts, tokens, vector store stats)
  * `/reindex`: Rescan the directory and ingest new/updated documents
  * `/history`: Display the full conversation history for the current session
  * `/config`: Show the agent's current configuration
  * `/clear`: Clear the console screen
  * `/help`: Display all available commands

### Adding Knowledge

To give your agent new knowledge:
1. Add or modify `.md` or `.txt` files in your directory
2. Use the `/reindex` command to update the knowledge base
3. The agent will now have access to the new information

### The .jerry Folder

When you run Jerry in a directory, it creates a hidden `.jerry` folder containing:

```
your-project/
├── .jerry/                    # Hidden folder (won't clutter your workspace)
│   ├── .chroma/              # Vector database (ChromaDB)
│   ├── logs/                 # Legacy text logs (optional)
│   ├── sessions/             # Archived conversation transcripts (Markdown)
│   ├── sessions.db           # SQLite: Session history and messages
│   ├── logs.db               # SQLite: Events, token usage, errors
│   └── config.json           # Agent configuration
├── your-file.md              # Your documents (indexed automatically)
├── another-doc.txt           # Text files also indexed
└── ... your other files
```

**Important**: Add `.jerry/` to your `.gitignore` to keep agent state out of version control!

## Command-Line Options

```bash
python -m jerry [DIRECTORY] [OPTIONS]

Arguments:
  DIRECTORY              Directory to instantiate agent in (defaults to current directory)

Options:
  -p, --provider TEXT    LLM provider: gemini, openai (auto-detected if not specified)
  -n, --name TEXT        Custom name for the agent (defaults to directory name)
  -s, --system-prompt    Custom system prompt for the agent
  -r, --reindex          Force reindexing of all documents on startup
  --help                 Show this message and exit
```

## Examples

```bash
# Use current directory
python -m jerry

# Specific directory
python -m jerry ~/Documents/my-notes

# With OpenAI provider
python -m jerry --provider openai

# Custom agent with specific prompt
python -m jerry --name "CodeHelper" --system-prompt "You are a expert Python developer"

# Force reindex on startup
python -m jerry --reindex
```

-----

## How It Works

### Directory-Based Architecture

Jerry operates on a simple principle: any directory can become an intelligent workspace. When you run Jerry:

1. **Initialization**: Creates a `.jerry/` folder in your target directory
2. **Document Scanning**: Recursively finds all `.md` and `.txt` files (excluding `.jerry/`)
3. **Vectorization**: Chunks and embeds documents into a ChromaDB vector store
4. **RAG Pipeline**: When you ask questions, semantically relevant chunks are retrieved and used to augment the LLM's context
5. **Clean State**: All logs, configs, and data stored in `.jerry/` - your workspace stays clean

### The Agent Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Your Directory                        │
│  ┌────────────────────────────────────────────────┐     │
│  │  .jerry/                 (Hidden State)        │     │
│  │  ├── .chroma/           (Vector DB)            │     │
│  │  ├── logs/              (Agent logs)           │     │
│  │  ├── sessions/          (Conversation archive) │     │
│  │  ├── config.json        (Settings)             │     │
│  │  └── agent_state.json   (Session memory)       │     │
│  └────────────────────────────────────────────────┘     │
│                                                          │
│  📄 your-docs.md  ──────────┐                           │
│  📄 notes.txt      ──────────┤  Indexed & Retrieved     │
│  📁 subfolder/               │                           │
│     📄 more-docs.md ─────────┘                           │
└─────────────────────────────────────────────────────────┘
                         ↓
                   [LangGraph Agent]
                         ↓
              ┌─────────────────────┐
              │   LLM (Gemini/GPT)  │
              │   + Retrieved Docs   │
              │   + Tools            │
              └─────────────────────┘
```

### Agent Configuration

The `.jerry/config.json` file stores the agent's settings:

```json
{
  "name": "my-project-agent",
  "provider": "gemini",
  "system_prompt": "You are a helpful assistant specialized in this project.",
  "retrieve_top_k": 5,
  "tools": [
    "file_tools",
    "knowledge_tools",
    "web_search_tools"
  ]
}
```

You can manually edit this file to customize behavior, or it will be created automatically on first run.

## Technical Architecture

Jerry is built on modern AI frameworks:

- **LangChain**: Tool integration and document processing
- **LangGraph**: Agentic workflow orchestration
- **ChromaDB**: Vector store for semantic search
- **Rich**: Beautiful terminal interface
- **Typer**: CLI framework

### Key Components

- **AgentRuntime** (`core/agent_runtime.py`): Manages agent lifecycle, state, and orchestration
- **VectorStoreManager** (`core/vectorstore_manager.py`): Handles document indexing and retrieval
- **ToolProvider** (`core/tool_provider.py`): Provides sandboxed file operations and knowledge retrieval
- **GraphBuilder** (`core/graph_builder.py`): Constructs the LangGraph agent workflow
- **AgentMonitor** (`core/agent_monitor.py`): Logging and token usage tracking

## Contributing

Contributions are welcome! This project follows a clean architecture pattern that makes it easy to extend.

**Getting Started with Development:**
1. Fork the repository
2. Install in editable mode: `pip install -e ".[llms,vector]"`
3. Make your changes
4. Test with `jerry` command
5. Submit a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development setup, guidelines, and best practices.

**Ways to Contribute:**
- Add new tools in `core/tools/`
- Extend the graph logic in `core/graph_builder.py`
- Add new LLM providers
- Improve documentation
- Report bugs or suggest features

## License

[Your License Here]

## Acknowledgments

Built with LangChain, LangGraph, and ChromaDB.