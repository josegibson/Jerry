#!/usr/bin/env python3
"""
Simple CLI program to interact with multiple LLM providers (Gemini, OpenAI)
Entry point - imports and runs the main CLI application
"""
from dotenv import load_dotenv

# Load .env as early as possible
load_dotenv()

from core.cli import MultiLLMCLI


def main():
    """Entry point"""
    cli = MultiLLMCLI()
    cli.run()


if __name__ == "__main__":
    main()
