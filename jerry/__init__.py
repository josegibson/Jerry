"""
Jerry - AI Agent

A tool that enables users to instantiate an AI agent from any directory.
"""
__version__ = "0.1.0"

from pathlib import Path
import os
import dotenv

# Load environment variables from .env file at project root
# Walk up to find the .env file
_current = Path(__file__).parent
_root = _current.parent  # Jerry root (parent of jerry package)
_env_file = _root / ".env"
if _env_file.exists():
    dotenv.load_dotenv(_env_file)
