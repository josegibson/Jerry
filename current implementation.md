# Project Jerry: A Modular, Multi-Agent AI Framework

Jerry is a Python-based framework for building and managing multiple, independent AI agents. Each agent is a self-contained unit, living within its own directory, with its own knowledge base, tools, and configuration. The framework is powered by LangChain and LangGraph, providing a robust foundation for creating sophisticated, tool-using AI assistants.

## Key Features

*   **Directory-Scoped Agents**: Each agent is an independent entity defined by a directory.
*   **Ephemeral Sessions**: Each run is a new session, and the agent's conversational memory is cleared upon shutdown.
*   **Session Archiving**: Conversations are saved as Markdown transcripts.
*   **Retrieval-Augmented Generation (RAG)**: Agents can answer questions based on a private knowledge base.
*   **Extensible Tool System**: New capabilities can be easily added to agents.
*   **Pluggable Backends**: LLM providers and embedding models can be switched via environment variables.
*   **Structured Logging**: An `AgentMonitor` tracks token usage and logs all major events.

## Project Structure

The project is structured into several modules:

*   `core`: Contains the core logic for the agent runtime, graph builder, tool provider, and vector store manager.
*   `core/tools`: Defines the tools that agents can use, such as file system access, knowledge retrieval, and web search.
*   `core/tools/_legacy`: Contains legacy code for vault tools.
*   `planner`: A module for managing tasks.
*   `__main__.py`: The CLI for interacting with the agents.

## Detailed Summary

### `core` module

The `core` module is the heart of the Jerry framework. It contains the following key components:

*   **`AgentRuntime`**: This class manages the entire lifecycle of a directory-scoped agent. It assembles all necessary components, including the agent's configuration, state, knowledge base, and tools.
*   **`GraphBuilder`**: This class constructs the agent's conversational graph using LangGraph. It defines the nodes for retrieving information, calling the LLM, and executing tools.
*   **`ToolProvider`**: This class is a factory that constructs and provides a sandboxed, context-aware set of tools for a specific agent instance.
*   **`VectorStoreManager`**: This class manages all vector store operations for a single, sandboxed agent. It handles the loading, splitting, and indexing of documents, as well as the retrieval of information from the vector store.
*   **`AgentMonitor`**: This class is a comprehensive monitoring and logging utility for an agent's lifecycle. It tracks token usage and logs all major events to a structured JSON log file.

### `core/tools` module

The `core/tools` module defines the tools that agents can use. The tools are organized into the following categories:

*   **`file_tools`**: These tools provide a secure, workspace-aware interface for file operations, such as reading, writing, and listing files.
*   **`knowledge_tools`**: This tool allows agents to search their knowledge base for information.
*   **`web_search_tools`**: This tool allows agents to search the web for information.
*   **`planner_tools`**: These tools allow agents to interact with the `planner` module to manage tasks.

### `core/tools/_legacy` module

The `core/tools/_legacy` module contains legacy code for vault tools. This code is no longer used by the framework, but it is kept for reference.

### `planner` module

The `planner` module is a simple task management system. It allows agents to add, retrieve, and update tasks.

### `__main__.py`

The `__main__.py` file is the command-line interface (CLI) for interacting with the agents. It allows users to create, run, and manage agents.