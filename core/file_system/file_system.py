from __future__ import annotations
import json
from pathlib import Path

class FileSystem:
    """
    A system for interacting with the file system within a sandboxed workspace.
    """

    def __init__(self, workspace_path: Path):
        """
        Initializes the FileSystem with a specific workspace directory.

        Args:
            workspace_path: The root directory for all file operations.
        """
        self.workspace_path = workspace_path
        self.workspace_path.mkdir(parents=True, exist_ok=True)

    def _get_safe_path(self, rel_path: str) -> Path:
        """
        Resolves a relative path to ensure it's safely within the workspace.
        This is a critical security measure to prevent directory traversal attacks.
        """
        # Normalize path to prevent tricks like '..', './', or leading slashes.
        rel_path = rel_path.strip().lstrip('./').lstrip('/')
        safe_path = (self.workspace_path / rel_path).resolve()

        if not str(safe_path).startswith(str(self.workspace_path.resolve())):
            raise ValueError(
                f"Security Error: Path '{rel_path}' attempts to access files "
                "outside the designated agent workspace."
            )
        return safe_path

    def writeFile(self, path: str, data: str) -> str:
        """
        Writes or overwrites a file with the given content in the agent's workspace.
        """
        try:
            safe_path = self._get_safe_path(path)
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            safe_path.write_text(data, encoding='utf-8')
            return f"Successfully wrote to file '{path}'."
        except Exception as e:
            return json.dumps({"error": {"type": "WriteError", "message": str(e)}})

    def readFile(self, path: str) -> str:
        """
        Reads the entire content of a specified file from the agent's workspace.
        """
        try:
            safe_path = self._get_safe_path(path)
            if not safe_path.is_file():
                return json.dumps({"error": {"type": "FileNotFound", "message": f"The file '{path}' does not exist."}})
            return safe_path.read_text(encoding='utf-8')
        except Exception as e:
            return json.dumps({"error": {"type": "ReadError", "message": str(e)}})
