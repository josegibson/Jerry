"""
Read-related tools and schemas for the vault.
Includes reading a specific file from the vault.
"""

import subprocess
from typing import List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from ._utils import (
    get_vault_path,
    validate_path_in_vault,
    check_tool_available,
    read_file_python,
)


class VaultReadInput(BaseModel):
    """Input schema for reading vault files."""

    path: str = Field(description="Path to the file to read (relative to vault root)")
    pager: bool = Field(
        default=False,
        description="Whether to use pager for output (usually False for LLM use)",
    )

    @field_validator("path")
    def validate_path(cls, v):
        if not v or not v.strip():
            raise ValueError("Path cannot be empty")
        # Prevent accidental extension patterns like '*.md' – must be a concrete path
        if any(c in v for c in "*?[]"):
            raise ValueError("Path must be a concrete file name, not a glob pattern")
        return v.strip()


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

READ_TOOLS = [
    vault_read_file,
]