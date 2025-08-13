# Jerry

A LangGraph-based personal assistant with multi-LLM support.

## Obsidian Vault RAG

- Set `VAULT_PATH` to your Obsidian vault directory.
- Optional envs:
  - `CHROMA_PERSIST_DIR` (default: `.chroma`)
  - `CHROMA_COLLECTION` (default: `jerry_vault`)
  - `RETRIEVE_TOP_K` (default: `5`)
  - `OPENAI_API_KEY` and `OPENAI_EMBED_MODEL` for OpenAI embeddings, else uses SentenceTransformers (`all-MiniLM-L6-v2`).

### Build/Refresh the index

```bash
python -m vault_tools.cli reindex --persist-dir .chroma --chunk-size 1200 --chunk-overlap 200
```

Then run `jerry.py`. The graph now retrieves relevant vault chunks before invoking the LLM.