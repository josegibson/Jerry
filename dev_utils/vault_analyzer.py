# vault_cost_comparator.py

import os
import argparse
import tiktoken

# A dictionary of various embedding providers and their models/pricing.
# Prices are in USD per 1,000 tokens.
# Last verified: August 2025. Always double-check provider websites for the latest pricing.
EMBEDDING_PROVIDERS = {
    "OpenAI": {
        "text-embedding-3-small": {
            "price_per_1k_tokens": 0.00002,
            "notes": "High performance, cost-effective."
        },
        "text-embedding-3-large": {
            "price_per_1k_tokens": 0.00013,
            "notes": "Highest performance, larger vector size."
        }
    },
    "Google (Vertex AI)": {
        "text-embedding-004": {
            "price_per_1k_tokens": 0.00002,
            "notes": "Comparable to OpenAI's small model."
        },
         "text-multilingual-embedding-002": {
            "price_per_1k_tokens": 0.00002,
            "notes": "Optimized for multilingual use cases."
        }
    },
    "Cohere": {
        "embed-english-v3.0": {
            "price_per_1k_tokens": 0.0001,
            "notes": "Designed for search & retrieval (RAG)."
        },
        "embed-multilingual-v3.0": {
            "price_per_1k_tokens": 0.0002,
            "notes": "Supports over 100 languages."
        }
    },
    "Amazon (AWS Bedrock)": {
        "amazon.titan-embed-text-v2": {
            "price_per_1k_tokens": 0.00002,
            "notes": "AWS's new, cost-effective embedding model."
        },
        "cohere.embed-english-v3": {
            "price_per_1k_tokens": 0.0001,
            "notes": "Cohere model hosted on AWS."
        }
    },
    "Together AI": {
        "mixedbread-ai/mxbai-embed-large-v1": {
            "price_per_1k_tokens": 0.0001,
            "notes": "Popular high-performance open model."
        }
    }
}

# Use OpenAI's tokenizer as a general standard for estimation.
TOKENIZER_ENCODING = "cl100k_base"

def count_tokens_in_vault(vault_path: str) -> (int, int):
    """
    Traverses an Obsidian vault, counts markdown files and their total tokens.
    """
    if not os.path.isdir(vault_path):
        print(f"Error: Directory not found at '{vault_path}'")
        return 0, 0

    try:
        encoding = tiktoken.get_encoding(TOKENIZER_ENCODING)
    except ValueError:
        print(f"Error: Could not load tokenizer encoding '{TOKENIZER_ENCODING}'.")
        return 0, 0
        
    total_tokens = 0
    md_file_count = 0
    print(f"🔍 Analyzing vault at: {os.path.abspath(vault_path)}\n")

    for root, _, files in os.walk(vault_path):
        for file in files:
            if file.endswith(".md"):
                md_file_count += 1
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        total_tokens += len(encoding.encode(content))
                except Exception as e:
                    print(f"Could not read or process file {file_path}: {e}")

    return total_tokens, md_file_count

def print_cost_comparison(total_tokens: int, file_count: int):
    """Prints a formatted table of embedding costs across different providers."""
    print("---" * 20)
    print("📊 VAULT ANALYSIS")
    print("---" * 20)
    print(f"Total Markdown Files Found: {file_count:,}")
    print(f"Total Tokens Estimated:     {total_tokens:,}")
    print("---" * 20)
    print("\n💰 EMBEDDING COST COMPARISON (ONE-TIME COST)")
    print("---" * 20)
    print(f"{'Provider':<22} {'Model':<35} {'Est. Cost (USD)'}")
    print(f"{'-'*20:<22} {'-'*33:<35} {'-'*17}")

    for provider, models in EMBEDDING_PROVIDERS.items():
        for model_name, details in models.items():
            cost = (total_tokens / 1000) * details['price_per_1k_tokens']
            print(f"{provider:<22} {model_name:<35} ${cost:<17.6f}")

    print("---" * 20)
    print("\nDisclaimer: Prices are for illustrative purposes. Always verify with the provider.")

def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(
        description="Count tokens and compare embedding costs for an Obsidian vault."
    )
    parser.add_argument(
        "vault_path", 
        type=str, 
        help="The path to your Obsidian vault directory."
    )
    args = parser.parse_args()

    total_tokens, file_count = count_tokens_in_vault(args.vault_path)

    if file_count > 0:
        print_cost_comparison(total_tokens, file_count)

if __name__ == "__main__":
    main()