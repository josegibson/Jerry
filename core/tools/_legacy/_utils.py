"""
Shared utilities for vault tools and CLI.

This module centralizes path validation, environment handling, external tool
checks, and pure-Python fallbacks for list/search/read so both the CLI and
LangChain tools can use the same, Windows-friendly logic.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Iterator, List, Optional, Tuple


# ----------------------------- Env and path helpers -----------------------------

def get_vault_path() -> Path:
    """Resolve and validate the vault root directory from VAULT_PATH.

    Raises
    ------
    ValueError
        If the environment variable is not set or the path is invalid.
    """
    vault_path = os.getenv("VAULT_PATH")
    if not vault_path:
        raise ValueError("VAULT_PATH environment variable not set")

    path = Path(vault_path)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Vault path '{vault_path}' does not exist or is not a directory")

    return path.resolve()


def validate_path_in_vault(file_path: str, vault_path: Path) -> Path:
    """Ensure a target path is within the vault root.

    Parameters
    ----------
    file_path: str
        Relative or absolute path to a file that should be inside the vault.
    vault_path: Path
        The vault root path.

    Returns
    -------
    Path
        The resolved absolute path if validation passes.

    Raises
    ------
    ValueError
        If the path escapes the vault root.
    """
    target_path = Path(file_path)

    if not target_path.is_absolute():
        target_path = vault_path / target_path

    target_path = target_path.resolve()

    try:
        target_path.relative_to(vault_path)
    except ValueError:
        raise ValueError(f"Path '{file_path}' is not within vault directory")

    return target_path


# ----------------------------- External tools ----------------------------------

def check_tool_available(tool: str) -> bool:
    """Return True if an external command is available in PATH."""
    try:
        subprocess.run([tool, "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# ----------------------------- Hidden file helpers -----------------------------

def _is_hidden_path(path: Path) -> bool:
    """Simple hidden check: treat dot-prefixed parts as hidden (cross-platform)."""
    return any(part.startswith(".") for part in path.parts)


# ----------------------------- Pure-Python list/search -------------------------

def iter_vault_files(vault_path: Path, ext: Optional[str] = "md", include_hidden: bool = False) -> Iterator[Path]:
    """Iterate files under vault, optionally filtering by extension and hidden flag."""
    normalized_ext = (ext or "").lower().lstrip(".")
    for path in vault_path.rglob("*"):
        if not path.is_file():
            continue
        if normalized_ext and path.suffix.lower().lstrip(".") != normalized_ext:
            continue
        if not include_hidden and _is_hidden_path(path.relative_to(vault_path)):
            continue
        yield path


def list_files_python(ext: Optional[str] = "md", hidden: bool = False) -> List[str]:
    """Return a list of file paths (as strings) within the vault using pure Python."""
    vault_path = get_vault_path()
    return [str(p) for p in iter_vault_files(vault_path, ext=ext, include_hidden=hidden)]


def search_content_python(
    term: str,
    ignore_case: bool = False,
    max_count: Optional[int] = None,
    file_ext: str = "md",
) -> List[str]:
    """Search vault files for `term` using pure Python.

    Returns a list of matches formatted like: "path:line_number: line content".
    """
    vault_path = get_vault_path()

    pattern = re.compile(re.escape(term), re.IGNORECASE if ignore_case else 0)

    matches: List[str] = []
    for file_path in iter_vault_files(vault_path, ext=file_ext, include_hidden=False):
        per_file_count = 0
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                for idx, line in enumerate(f, start=1):
                    if pattern.search(line):
                        matches.append(f"{file_path}:{idx}: {line.rstrip()}")
                        per_file_count += 1
                        if max_count is not None and per_file_count >= max_count:
                            break
        except Exception:
            # Skip unreadable files
            continue

    return matches


def read_file_python(path: str) -> Tuple[Optional[str], Optional[str]]:
    """Read a file inside the vault and return (content, error).

    If an error occurs, returns (None, error_message).
    """
    try:
        vault_path = get_vault_path()
        file_path = validate_path_in_vault(path, vault_path)

        if not file_path.exists():
            return None, f"Error: File '{path}' does not exist"
        if not file_path.is_file():
            return None, f"Error: '{path}' is not a file"

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read(), None
    except Exception as e:
        return None, f"Error reading file: {e}" 