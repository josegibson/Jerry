"""
Write-related tools for the vault.
"""

import os
from pathlib import Path
from typing import List

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from ._utils import get_vault_path


class WriteMarkdownInput(BaseModel):
    """Input schema for writing markdown files."""

    filename: str = Field(description="Name of the file (should end with .md)")
    content: str = Field(description="The markdown content to write")
    directory: str = Field(
        default=".",
        description=(
            "Optional relative subdirectory. If omitted, writes to the current working area."
        ),
    )

    @field_validator("filename")
    def validate_filename(cls, v):
        if not v or not v.strip():
            raise ValueError("Filename cannot be empty")
        name = v.strip()
        # Very basic path safety: do not allow traversal via filename
        if any(p in name for p in ("..", "\\", "/")):
            raise ValueError("Filename must not contain path separators or '..'")
        return name

    @field_validator("directory")
    def normalize_directory(cls, v):
        v = (v or ".").strip()
        # Normalize common separators and strip any leading slashes to keep it relative
        v = v.replace("\\", "/").lstrip("/")
        # Prevent traversal in directory
        if ".." in v:
            raise ValueError("Directory must not contain '..'")
        return v if v else "."


@tool(args_schema=WriteMarkdownInput)
def write_markdown_file(filename: str, content: str, directory: str = ".") -> str:
    """
    Create or overwrite a markdown (.md) file with the given content.

    All writes are confined to the allowed working area. The `directory` argument
    is treated as a relative path within this area, and any attempt to escape is blocked.
    """
    vault_root = get_vault_path()

    # Define confined working root and ensure it exists
    agent_root = (vault_root / "jerry").resolve()
    os.makedirs(agent_root, exist_ok=True)

    # Ensure .md extension
    if not filename.endswith(".md"):
        filename = f"{filename}.md"

    # Resolve target directory under working root
    safe_dir = Path(directory.replace("\\", "/").lstrip("/")) if directory else Path(".")
    target_dir = (agent_root / safe_dir).resolve()

    # Ensure the target directory stays within the working root
    try:
        target_dir.relative_to(agent_root)
    except ValueError:
        return "Error: Target directory is not allowed"

    # Create directories as needed
    os.makedirs(target_dir, exist_ok=True)

    target_path = (target_dir / filename).resolve()

    # Safety: ensure final file path is still within the working root
    try:
        target_path.relative_to(agent_root)
    except ValueError:
        return "Error: Target file path is not allowed"

    # Write file
    try:
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        return f"Error writing file: {e}"

    # Return a path relative to the working root to avoid exposing absolute locations
    relative_path = target_path.relative_to(agent_root)
    return f"Markdown file written: {relative_path.as_posix()}"


# Export list of write tools
WRITE_TOOLS: List = [write_markdown_file] 