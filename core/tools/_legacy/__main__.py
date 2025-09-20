#!/usr/bin/env python3
"""
Entry point for vault_tools package.
Allows running: python -m vault_tools
"""

from dotenv import load_dotenv

# Load .env so VAULT_PATH and others are available
load_dotenv()

from .cli import main

if __name__ == "__main__":
    main()