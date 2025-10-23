"""
Core module for Jerry Agent Runtime.

This module provides the core business logic for the Jerry agent system.
For CLI interface, use `python -m jerry` or the `jerry` command.

Direct usage of core module is discouraged - use the jerry package instead.
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    print("❌ Please use 'jerry' command or 'python -m jerry' instead.")
    print("\nExamples:")
    print("  jerry .                    # Run in current directory")
    print("  jerry /path/to/project     # Run in specific directory")
    print("  jerry --help               # Show all options")
    sys.exit(1)

