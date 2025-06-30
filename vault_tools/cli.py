#!/usr/bin/env python3
"""
Vault Tools CLI - Command-line interface for vault file management.

Provides three main commands:
- list: List markdown files in the vault
- search: Search for text within vault files  
- read: Display vault files with syntax highlighting
"""

import subprocess
from typing import Optional

import typer
from typer import Typer, Option, Argument

from .utils import (
    get_vault_path,
    validate_path_in_vault,
    check_tool_available,
    list_files_python,
    search_content_python,
    read_file_python,
)


app = Typer(name="vault-tools", help="CLI tools for managing vault files")


@app.command(name="list")
def list_files(
    ext: Optional[str] = Option(None, "--ext", "-e", help="Filter by file extension (e.g., 'md')"),
    hidden: bool = Option(False, "--hidden", help="Include hidden files"),
) -> None:
    """List markdown files in the vault.

    Uses `fd` when available, otherwise falls back to a pure-Python traversal.
    """
    # Prefer fd if available for speed
    if check_tool_available("fd"):
        vault_path = get_vault_path()
        cmd = ["fd"]
        if ext:
            cmd.extend(["-e", ext])
        else:
            cmd.extend(["-e", "md"])
        if hidden:
            cmd.append("--hidden")
        cmd.extend([".", str(vault_path)])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            typer.echo(result.stdout)
            return
        except subprocess.CalledProcessError as e:
            typer.echo(f"Error running fd: {e.stderr}", err=True)
            raise typer.Exit(e.returncode)

    # Fallback to Python
    files = list_files_python(ext or "md", hidden)
    if files:
        typer.echo("\n".join(files))
    else:
        typer.echo(f"No {ext or 'md'} files found in vault")


@app.command()
def search(
    term: str = Argument(..., help="Search term to look for"),
    ignore_case: bool = Option(False, "--ignore-case", "-i", help="Case insensitive search"),
    max_count: Optional[int] = Option(None, "--max-count", "-m", help="Maximum number of matches per file"),
) -> None:
    """Search for text in vault markdown files.

    Uses `ripgrep` when available, otherwise falls back to a pure-Python search.
    """
    # Prefer ripgrep if available
    if check_tool_available("rg"):
        vault_path = get_vault_path()
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
            if result.stdout:
                typer.echo(result.stdout)
            else:
                typer.echo("No matches found.")
            return
        except subprocess.CalledProcessError as e:
            if e.returncode == 1:
                typer.echo("No matches found.")
                return
            typer.echo(f"Error running ripgrep: {e.stderr}", err=True)
            raise typer.Exit(e.returncode)

    # Fallback to Python
    matches = search_content_python(term, ignore_case=ignore_case, max_count=max_count, file_ext="md")
    if matches:
        typer.echo("\n".join(matches))
    else:
        typer.echo("No matches found.")


@app.command()
def read(
    path: str = Argument(..., help="Path to file to read"),
    pager: bool = Option(True, "--pager/--no-pager", help="Use pager for output"),
) -> None:
    """Display vault file, preferring `bat` when available.

    Falls back to plain text output when `bat` is not installed.
    """
    # If bat is available, use it for nice output
    if check_tool_available("bat"):
        vault_path = get_vault_path()
        file_path = validate_path_in_vault(path, vault_path)
        if not file_path.exists():
            typer.echo(f"Error: File '{path}' does not exist", err=True)
            raise typer.Exit(1)
        if not file_path.is_file():
            typer.echo(f"Error: '{path}' is not a file", err=True)
            raise typer.Exit(1)
        cmd = ["bat"]
        if not pager:
            cmd.append("--paging=never")
        cmd.append(str(file_path))
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as e:
            typer.echo(f"Error running bat: {e}", err=True)
            raise typer.Exit(e.returncode)

    # Fallback to Python plain read
    content, error = read_file_python(path)
    if error:
        typer.echo(error, err=True)
        raise typer.Exit(1)
    typer.echo(content)


def main():
    """Entry point for the vault-tools CLI."""
    app()


if __name__ == "__main__":
    main() 