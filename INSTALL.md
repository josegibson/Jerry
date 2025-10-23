# Jerry Installation Guide

## Quick Install (Recommended)

### 1. Prerequisites
- Python 3.9 or higher
- pip package manager

### 2. Install Jerry

```bash
# Clone the repository
git clone https://github.com/josegibson/Jerry.git
cd Jerry

# Install with pip
pip install -e .
```

That's it! The `jerry` command is now available globally.

### 3. Configure API Keys

Create a `.env` file in your project root:

```bash
# Required: At least one LLM provider
GOOGLE_API_KEY="your-google-api-key-here"
OPENAI_API_KEY="your-openai-api-key-here"

# Optional: Web search (for web_search tool)
TAVILY_API_KEY="your-tavily-api-key-here"
```

### 4. Run Jerry

```bash
# Navigate to any directory with documents
cd /path/to/your/project

# Start Jerry
jerry
```

Jerry will automatically:
- Create a `.jerry` folder for agent state
- Index all `.md` and `.txt` files
- Start an interactive chat session

---

## Alternative: Virtual Environment

If you prefer isolated environments:

```bash
# Create virtual environment
python -m venv jerry_env

# Activate it
# Windows:
jerry_env\Scripts\activate
# macOS/Linux:
source jerry_env/bin/activate

# Install Jerry
pip install -e .
```

---

## Verify Installation

```bash
# Check Jerry is installed
jerry --help

# Or run validation script
python validate_install.py
```

---

## Troubleshooting

### Command not found: jerry
Make sure pip's scripts directory is in your PATH:
- Windows: `%APPDATA%\Python\Scripts`
- macOS/Linux: `~/.local/bin`

### Import errors
Reinstall dependencies:
```bash
pip install --upgrade pip
pip install -e . --force-reinstall
```

### API key issues
Verify your `.env` file is in the correct location and contains valid keys.

---

## Next Steps

- Read [QUICKSTART.md](docs/QUICKSTART.md) for usage examples
- See [USER_GUIDE.md](docs/USER_GUIDE.md) for detailed features
- Check [CONTRIBUTING.md](docs/CONTRIBUTING.md) to contribute
