#!/usr/bin/env python3
"""
Vault Tools CLI - Command-line interface for vault file management.

Provides three main commands:
- list: List markdown files in the vault
- search: Search for text within vault files  
- read: Display vault files with syntax highlighting
- write: Create or overwrite a markdown file in the vault
"""

import subprocess
import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from typer import Typer, Option, Argument

from ._utils import (
    get_vault_path,
    validate_path_in_vault,
    check_tool_available,
    list_files_python,
    search_content_python,
    read_file_python,
)
from .vectorstore import ensure_vector_index, semantic_search_direct
from .vault_write_tools import write_markdown_file
from .vault_read_tools import vault_read_file
from .vault_search_tools import vault_list_files as tool_vault_list_files, vault_search_content as tool_vault_search_content, web_search as tool_web_search, semantic_vault_search as tool_semantic_vault_search

import chromadb

app = Typer(name="vault-tools", help="CLI tools for managing vault files")


@app.command(name="list")
def list_files(
    ext: Optional[str] = Option(None, "--ext", "-e", help="Filter by file extension (e.g., 'md')"),
    hidden: bool = Option(False, "--hidden", help="Include hidden files"),
) -> None:
    """List markdown files in the vault.

    Uses the same implementation as the LangChain tool.
    """
    result = tool_vault_list_files.invoke({
        "ext": (ext or "md"),
        "hidden": hidden,
    })
    typer.echo(result)


@app.command()
def search(
    term: str = Argument(..., help="Search term to look for"),
    ignore_case: bool = Option(False, "--ignore-case", "-i", help="Case insensitive search"),
    max_count: Optional[int] = Option(None, "--max-count", "-m", help="Maximum number of matches per file"),
) -> None:
    """Search for text in vault markdown files.

    Uses the same implementation as the LangChain tool.
    """
    result = tool_vault_search_content.invoke({
        "term": term,
        "ignore_case": ignore_case,
        "max_count": max_count,
    })
    typer.echo(result)


@app.command()
def read(
    path: str = Argument(..., help="Path to file to read"),
    pager: bool = Option(False, "--pager/--no-pager", help="Use pager for output when available"),
) -> None:
    """Display vault file via the shared read tool.

    Prefers `bat` when available, falls back to plain text.
    """
    result = vault_read_file.invoke({
        "path": path,
        "pager": pager,
    })
    typer.echo(result)


@app.command()
def write(
    filename: str = Argument(..., help="Name of the markdown file to write ('.md' appended if missing)"),
    content: str = Argument(..., help="Markdown content to write"),
    directory: str = Option("./jerry", "--directory", "-d", help="Subdirectory under the vault"),
) -> None:
    """Create or overwrite a markdown file inside the vault."""
    result = write_markdown_file.invoke({
        "filename": filename,
        "content": content,
        "directory": directory,
    })
    # write_markdown_file returns a string
    typer.echo(result)


@app.command()
def web_search(
    query: str = Argument(..., help="Web search query"),
    max_results: int = Option(5, "--max-results", "-n", help="Max results to return (1-10)"),
) -> None:
    """Search the web and return a concise set of results with titles and URLs."""
    result = tool_web_search.invoke({
        "query": query,
        "max_results": max_results,
    })
    typer.echo(result)


@app.command()
def semantic_vault_search(
    query: str = Argument(..., help="Semantic search query against the vault index"),
    k: int = Option(5, "--k", "-k", help="Number of results (1-10)"),
) -> None:
    """Semantic search over the Obsidian vault vector store (LangChain retriever)."""
    result = tool_semantic_vault_search.invoke({
        "query": query,
        "k": k,
    })
    typer.echo(result)


@app.command()
def semantic_search(
    query: str = typer.Argument(..., help="Semantic search query"),
    k: int = typer.Option(5, "--k", "-k", help="Number of results"),
) -> None:
    """Semantic search over the persisted Chroma vector store without LangChain retriever."""
    load_dotenv()

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", ".chroma")
    results = semantic_search_direct(query=query, k=k, persist_directory=persist_dir)

    if results:
        for i, r in enumerate(results, start=1):
            typer.echo(f"[{i}] Source: {r['source']}\n{r['content']}\nScore: {r['score']}\n{'-'*40}\n")
    else:
        typer.echo("No results found.")




@app.command()
def reindex(
    vault_path: Optional[str] = typer.Option(None, help="Path to Obsidian vault. Defaults to VAULT_PATH"),
    persist_dir: str = typer.Option(".chroma", help="Directory to persist Chroma DB"),
    chunk_size: int = typer.Option(1200, help="Chunk size (characters)"),
    chunk_overlap: int = typer.Option(200, help="Chunk overlap (characters)"),
    glob: str = typer.Option("**/*.md", help="Glob for files to index"),
):
    """(Re)build the local vector index from the vault."""
    load_dotenv()
    if vault_path:
        os.environ["VAULT_PATH"] = vault_path
    if not os.getenv("VAULT_PATH"):
        raise typer.Exit(code=2)

    index = ensure_vector_index(
        persist_directory=persist_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        file_glob=glob,
    )
    stats = index.get("_stats", {}) if isinstance(index, dict) else {}
    typer.echo(f"Index ready at {persist_dir}. {stats}")



def main():
    """Entry point for the vault-tools CLI."""
    app()


if __name__ == "__main__":
    main() 