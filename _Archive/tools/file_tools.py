from __future__ import annotations

import json
from pathlib import Path
from typing import List

from langchain_core.tools import BaseTool, tool


def create_file_tools(workspace_path: Path) -> List[BaseTool]:
    """
    Factory function to create a suite of secure, workspace-aware file tools.

    Args:
        workspace_path: The root directory for the agent's sandboxed file operations.

    Returns:
        A list of configured file management tools.
    """
    # Ensure the workspace directory exists.
    workspace_path.mkdir(parents=True, exist_ok=True)

    # --- Helper Functions (scoped to the factory) ---

    def _get_safe_path(rel_path: str) -> Path:
        """
        Resolves a relative path to ensure it's safely within the workspace.
        This is a critical security measure to prevent directory traversal attacks.
        """
        # Normalize path to prevent tricks like '..', './', or leading slashes.
        rel_path = rel_path.strip().lstrip('./').lstrip('/')
        safe_path = (workspace_path / rel_path).resolve()

        if not str(safe_path).startswith(str(workspace_path.resolve())):
            raise ValueError(
                f"Security Error: Path '{rel_path}' attempts to access files "
                "outside the designated agent workspace."
            )
        return safe_path

    def _format_error(error_type: str, message: str) -> str:
        """Helper to create a standardized JSON error string."""
        return json.dumps({"error": {"type": error_type, "message": message}})

    # --- Tool Definitions (using helpers and captured workspace_path) ---

    @tool
    def write_file(path: str, content: str) -> str:
        """Writes or overwrites a file with the given content in the agent's workspace."""
        try:
            safe_path = _get_safe_path(path)
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            safe_path.write_text(content, encoding='utf-8')
            return f"Successfully wrote to file '{path}'."
        except Exception as e:
            return _format_error("WriteError", str(e))

    @tool
    def read_file(path: str) -> str:
        """Reads the entire content of a specified file from the agent's workspace."""
        try:
            safe_path = _get_safe_path(path)
            if not safe_path.is_file():
                return f'{{"error": "FileNotFound", "message": "The file \'{path}\' does not exist."}}'
            return safe_path.read_text(encoding='utf-8')
        except Exception as e:
            return _format_error("ReadError", str(e))

    @tool
    def list_files(directory: str = ".") -> str:
        """Lists all files and directories in a specified subdirectory of the workspace."""
        try:
            target_dir = _get_safe_path(directory)
            if not target_dir.is_dir():
                return f'{{"error": "DirectoryNotFound", "message": "The directory \'{directory}\' does not exist."}}'

            items = sorted([f.name for f in target_dir.iterdir()])
            return json.dumps(items) if items else "[]"
        except Exception as e:
            return _format_error("ListError", str(e))

    return [write_file, read_file, list_files]
