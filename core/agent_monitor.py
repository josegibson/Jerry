from __future__ import annotations

import os
import json
import logging
import tiktoken
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# A dictionary of various embedding providers and their models/pricing.
# Prices are in USD per 1,000 tokens.
# Last verified: September 2025.
EMBEDDING_PRICING = {
    "OpenAI:text-embedding-3-small": 0.00002,
    "OpenAI:text-embedding-3-large": 0.00013,
    "HuggingFace:all-MiniLM-L6-v2": 0.0, # Local models are free
}

# Use OpenAI's tokenizer as a general standard for estimation.
TOKENIZER_ENCODING = "cl100k_base"

class AgentMonitor:
    """
    A comprehensive monitoring and logging utility for an agent's lifecycle.

    This class handles:
    - Structured logging of agent operations to a dedicated log file.
    - Detailed token usage tracking, including session and provider-specific metrics.
    - Analysis of the agent's knowledge base (workspace files and vector store).
    """

    def __init__(self, agent_name: str, log_dir: Path):
        self.agent_name = agent_name
        self.log_file_path = log_dir / f"agent_{agent_name}.log"
        self._configure_logging()
        self.logger = logging.getLogger(self.agent_name)

        self.token_metrics: Dict[str, Any] = {
            "session_total": {"input": 0, "output": 0, "total": 0},
            "by_provider": {}
        }

    def _configure_logging(self):
        """Sets up a dedicated logger for the agent."""
        # Prevent logs from propagating to the root logger
        logger = logging.getLogger(self.agent_name)
        if logger.hasHandlers():
            logger.handlers.clear()
            
        logger.setLevel(logging.DEBUG)
        
        # Create a file handler for structured logging
        handler = logging.FileHandler(self.log_file_path, mode='a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Logs a structured event."""
        log_entry = {"event": event_type, **data}
        self.logger.info(json.dumps(log_entry))

    def track_token_usage(self, provider_name: str, usage_data: Dict[str, int]):
        """Updates token metrics for the session and by provider."""
        input_tokens = usage_data.get("input_tokens", 0)
        output_tokens = usage_data.get("output_tokens", 0)
        total_tokens = usage_data.get("total_tokens", input_tokens + output_tokens)

        # Update session total
        self.token_metrics["session_total"]["input"] += input_tokens
        self.token_metrics["session_total"]["output"] += output_tokens
        self.token_metrics["session_total"]["total"] += total_tokens

        # Update per-provider stats
        provider_stats = self.token_metrics["by_provider"].setdefault(provider_name, {"input": 0, "output": 0, "total": 0})
        provider_stats["input"] += input_tokens
        provider_stats["output"] += output_tokens
        provider_stats["total"] += total_tokens
        
        self.log_event("token_usage", {
            "provider": provider_name, 
            "last_turn": {"input": input_tokens, "output": output_tokens},
            "session_total": self.token_metrics["session_total"]
        })

    def get_knowledge_base_analysis(self, workspace_path: Path, vector_store_manager: Any) -> Dict[str, Any]:
        """
        Analyzes the agent's workspace and vector store, integrating logic
        from the original vault_analyzer.py.
        """
        try:
            encoding = tiktoken.get_encoding(TOKENIZER_ENCODING)
        except Exception as e:
            return {"error": f"Could not load tokenizer: {e}"}

        # 1. Analyze workspace files
        total_tokens = 0
        file_count = 0
        for file_path in workspace_path.rglob("*.md"):
            if file_path.is_file():
                file_count += 1
                try:
                    content = file_path.read_text(encoding='utf-8')
                    total_tokens += len(encoding.encode(content))
                except Exception:
                    continue # Skip files that can't be read

        # 2. Get vector store metadata
        embedding_model = vector_store_manager.embedding_model_name
        num_chunks = vector_store_manager.db._collection.count()
        
        # 3. Estimate embedding cost
        cost_per_1k = EMBEDDING_PRICING.get(embedding_model, 0.0)
        estimated_cost = (total_tokens / 1000) * cost_per_1k
        
        analysis = {
            "workspace_stats": {
                "file_count": file_count,
                "total_tokens": total_tokens,
            },
            "vector_store_stats": {
                "embedding_model": embedding_model,
                "indexed_chunks": num_chunks,
                "estimated_one_time_cost_usd": f"{estimated_cost:.6f}"
            }
        }
        self.log_event("knowledge_base_analysis", analysis)
        return analysis

    def get_token_metrics(self) -> Dict[str, Any]:
        """Returns the current token usage metrics."""
        return self.token_metrics
