# Multi-LLM CLI Chat

A LangGraph-native command-line interface to interact with multiple LLM providers (OpenAI, Gemini) with vault tools.

## Vault Tools

This project includes `vault_tools` - a CLI utility for managing and searching markdown files in a vault directory, and LangChain-compatible tools the LLM can call.

## Features

### LLM Chat (LangGraph)
✅ **LangGraph-native** orchestration with a single conversation state  
✅ **Multiple Backends** - OpenAI and Gemini via LangChain chat models  
✅ **Auto-Detection** - Picks a backend based on available API keys  
✅ **Tool Calling** - LLM can call `vault_list_files`, `vault_search_content`, `vault_read_file`

### Vault Tools
✅ **File Listing** - List markdown files in vault using `fd`  
✅ **Text Search** - Search content using `ripgrep`  
✅ **File Display** - View files with syntax highlighting using `bat`  
✅ **Path Validation** - Ensures files are within vault directory  
✅ **Tool Detection** - Checks external dependencies

## Setup

### Dependencies
```cmd
pip install -r requirements.txt
```

### API Keys (choose one or both)

**OpenAI**
```cmd
set OPENAI_API_KEY=your_openai_key_here
```
Get your key from: `https://platform.openai.com/api-keys`

**Gemini**
```cmd
set GEMINI_API_KEY=your_gemini_key_here
```
Get your key from: `https://makersuite.google.com/app/apikey`

## Usage

Run the program:
```cmd
python jerry.py
```

- Type your question/prompt and press Enter
- Type `quit`, `exit`, or `q` to exit

## Vault Tools Usage

The `vault_tools` module provides three main commands for managing markdown files in a vault directory.

### Prerequisites

Install the required external tools:

**Windows:**
```cmd
# Install via scoop (recommended)
scoop install fd ripgrep bat

# Or via chocolatey
choco install fd ripgrep bat
```

**macOS:**
```bash
brew install fd ripgrep bat
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt install fd-find ripgrep bat

# Or build from source
cargo install fd-find ripgrep bat
```

### Environment Setup

Set the vault path environment variable:
```cmd
# Windows
set VAULT_PATH=C:\path\to\your\vault

# Linux/macOS
export VAULT_PATH=/path/to/your/vault
```

### Commands

#### 1. List Files (`vault-tools list`)

List markdown files in the vault directory:

```cmd
python -m vault_tools list                    # List all .md files
python -m vault_tools list --ext txt          # List .txt files  
python -m vault_tools list --hidden           # Include hidden files
```

**Features:**
- Uses `fd` for fast file discovery
- Skips hidden files by default
- Respects `.gitignore` patterns
- Supports custom file extensions

#### 2. Search Content (`vault-tools search`)

Search for text within vault files:

```cmd
python -m vault_tools search "python"                    # Basic search
python -m vault_tools search "Python" --ignore-case     # Case insensitive
python -m vault_tools search "TODO" --max-count 5       # Limit matches per file
```

**Features:**
- Uses `ripgrep` for ultra-fast text search
- Returns file paths and line numbers
- Case-sensitive by default
- Optional match count limits

#### 3. Read Files (`vault-tools read`)

Display files with syntax highlighting:

```cmd
python -m vault_tools read notes/readme.md              # With pager
python -m vault_tools read notes/readme.md --no-pager   # Direct output
```

**Features:**
- Uses `bat` for syntax highlighting
- Optional pager support
- Path validation (must be within vault)
- Handles relative and absolute paths

### Security Features

- **Path Validation**: All file operations are restricted to the vault directory
- **Tool Verification**: Checks for external tool availability before execution
- **Error Handling**: Graceful handling of missing files, tools, or permissions

### Example Workflow

```cmd
# Set up vault
set VAULT_PATH=C:\Users\You\Documents\Notes

# List all markdown files
python -m vault_tools list

# Search for specific content
python -m vault_tools search "project ideas" --ignore-case

# Read a specific file
python -m vault_tools read projects/new-features.md
```