"""
Vault Tools for LLM Integration

This module aggregates LangChain-compatible tool definitions for vault operations
by importing them from dedicated modules (read, search, write). It preserves the
original API surface for backwards compatibility.
"""

from typing import List


from .vault_read_tools import READ_TOOLS
from .vault_write_tools import WRITE_TOOLS  # placeholder for future write tools
from .vault_search_tools import SEARCH_TOOLS

# List of all available vault tools for easy import
VAULT_TOOLS = [
    *SEARCH_TOOLS,
    *READ_TOOLS,
    *WRITE_TOOLS,
]


def get_vault_tools() -> List:
    """Get all vault tools for use with LangChain agents."""
    return VAULT_TOOLS


def get_vault_tools_description() -> str:
    """Get a description of all available vault tools."""
    return """
Available Vault Tools:
1. web_search: Search the web for relevant information
2. semantic_vault_search: Semantic search over the Obsidian vault vector store
3. vault_list_files: List files in the vault directory
4. vault_search_content: Search for text content within vault files  
5. vault_read_file: Read and display the contents of a vault file

These tools allow you to explore, search, and read content from the vault directory and the web.
Make sure VAULT_PATH environment variable is set to use these tools.
""" 