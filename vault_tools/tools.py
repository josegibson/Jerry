"""
Vault Tools for LLM Integration

This module provides LangChain-compatible tool definitions for vault operations
that can be used by LLM providers to interact with the vault programmatically.
"""

import subprocess
from typing import List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from .utils import (
    get_vault_path,
    validate_path_in_vault,
    check_tool_available,
    list_files_python,
    search_content_python,
    read_file_python,
)


# ---------------------------------------------------------------------------
# Input-validation helpers (industry-standard approach)
# • Validation is done in the Pydantic model so that a malformed tool call is
#   rejected before the function logic runs.  This mirrors common practice in
#   frameworks such as FastAPI or LangChain's recommended pattern for tools.
# • Where appropriate we *raise* a validation error instead of silently
#   coercing.  The runtime will feed that error back to the LLM so it can retry
#   with a conforming argument set – the same self-correction loop that OpenAI
#   and Google demonstrate in their examples.
# ---------------------------------------------------------------------------

# Allowed file-extensions for the vault listing/searching tools.  Add more as
# needed; keep them lower-case and **without** leading dots to simplify checks.
ALLOWED_EXTENSIONS = {
    "md",
    "txt",
    "json",
    "csv",
    "rst",
}


class VaultListInput(BaseModel):
    """Input schema for listing vault files."""
    ext: Optional[str] = Field(
        default="md", 
        description="File extension to filter by (e.g., 'md', 'txt')"
    )
    hidden: bool = Field(
        default=False, 
        description="Whether to include hidden files in the results"
    )

    # --- Validators ---------------------------------------------------------
    @field_validator("ext", mode="before")
    def normalise_and_validate_ext(cls, v):
        """Ensure extension is one of the allowed set; strip leading dot."""
        if v is None or v == "":
            return "md"
        v = v.lower().lstrip(".")
        if v not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported extension '{v}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            )
        return v


class VaultSearchInput(BaseModel):
    """Input schema for searching vault content."""
    term: str = Field(
        description="The search term to look for in vault files"
    )
    ignore_case: bool = Field(
        default=False, 
        description="Whether to perform case-insensitive search"
    )
    max_count: Optional[int] = Field(
        default=None, 
        description="Maximum number of matches per file"
    )

    @field_validator("term")
    def validate_term(cls, v):
        if not v or not v.strip():
            raise ValueError("Search term cannot be empty")
        if len(v.strip()) > 200:
            raise ValueError("Search term is too long (max 200 characters)")
        return v.strip()


class VaultReadInput(BaseModel):
    """Input schema for reading vault files."""
    path: str = Field(
        description="Path to the file to read (relative to vault root)"
    )
    pager: bool = Field(
        default=False, 
        description="Whether to use pager for output (usually False for LLM use)"
    )

    @field_validator("path")
    def validate_path(cls, v):
        if not v or not v.strip():
            raise ValueError("Path cannot be empty")
        # Prevent accidental extension patterns like '*.md' – must be a concrete path
        if any(c in v for c in "*?[]"):
            raise ValueError("Path must be a concrete file name, not a glob pattern")
        return v.strip()




@tool(args_schema=VaultListInput)
def vault_list_files(ext: str = "md", hidden: bool = False) -> str:
    """
    List files in the vault directory.

    Uses `fd` when available, otherwise falls back to a pure-Python traversal.
    """
    # Prefer fd if available for speed
    if check_tool_available("fd"):
        try:
            vault_path = get_vault_path()
        except ValueError as e:
            return f"Error: {e}"

        cmd = ["fd"]
        if ext:
            cmd.extend(["-e", ext])
        if hidden:
            cmd.append("--hidden")
        cmd.extend([".", str(vault_path)])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            file_list = result.stdout.strip()
            if not file_list:
                return f"No {ext} files found in vault"
            return file_list
        except subprocess.CalledProcessError as e:
            return f"Error running fd: {e.stderr}"

    # Fallback to Python
    try:
        files = list_files_python(ext, hidden)
        return "\n".join(files) if files else f"No {ext} files found in vault"
    except Exception as e:
        return f"Error listing files: {e}"


@tool(args_schema=VaultSearchInput)
def vault_search_content(term: str, ignore_case: bool = False, max_count: Optional[int] = None) -> str:
    """
    Search for text content within vault files.

    Uses `ripgrep` when available, otherwise falls back to a pure-Python search.
    """
    if check_tool_available("rg"):
        try:
            vault_path = get_vault_path()
        except ValueError as e:
            return f"Error: {e}"

        cmd = ["rg", "--line-number", "--with-filename"]
        if ignore_case:
            cmd.append("--ignore-case")
        if max_count:
            cmd.extend(["--max-count", str(max_count)])
        cmd.extend(["--type", "md"])  # md files only
        cmd.append(term)
        cmd.append(str(vault_path))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout.strip() if result.stdout else f"No matches found for '{term}'"
        except subprocess.CalledProcessError as e:
            if e.returncode == 1:
                return f"No matches found for '{term}'"
            return f"Error running ripgrep: {e.stderr}"

    # Fallback to Python
    try:
        matches = search_content_python(term, ignore_case=ignore_case, max_count=max_count, file_ext="md")
        return "\n".join(matches) if matches else f"No matches found for '{term}'"
    except Exception as e:
        return f"Error searching content: {e}"


@tool(args_schema=VaultReadInput)
def vault_read_file(path: str, pager: bool = False) -> str:
    """
    Read and display the contents of a vault file.
    
    This tool reads a specific file from the vault and returns its contents.
    It's useful for retrieving the full content of a specific document.
    
    Args:
        path: Path to the file to read (relative to vault root)
        pager: Whether to use pager for output (usually False for LLM use)
    
    Returns:
        The contents of the specified file
    """
    # If bat is available, prefer it for consistent formatting
    if check_tool_available("bat"):
        try:
            vault_path = get_vault_path()
            file_path = validate_path_in_vault(path, vault_path)
        except ValueError as e:
            return f"Error: {e}"

        if not file_path.exists():
            return f"Error: File '{path}' does not exist"
        if not file_path.is_file():
            return f"Error: '{path}' is not a file"

        cmd = ["bat"]
        if not pager:
            cmd.append("--paging=never")
        # Add plain text output for LLM consumption
        cmd.extend(["--style=plain", "--color=never"])
        cmd.append(str(file_path))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Error running bat: {e}"

    # Fallback to regular Python reading
    content, error = read_file_python(path)
    return content if content is not None else (error or "Error reading file")


# List of all available vault tools for easy import
VAULT_TOOLS = [
    vault_list_files,
    vault_search_content,
    vault_read_file,
]


def get_vault_tools() -> List:
    """Get all vault tools for use with LangChain agents."""
    return VAULT_TOOLS


def get_vault_tools_description() -> str:
    """Get a description of all available vault tools."""
    return """
Available Vault Tools:
1. vault_list_files: List files in the vault directory
2. vault_search_content: Search for text content within vault files  
3. vault_read_file: Read and display the contents of a vault file

These tools allow you to explore, search, and read content from the vault directory.
Make sure VAULT_PATH environment variable is set to use these tools.
""" 