"""
Embedding client for LM Studio using Qwen3 embedding models.

Responsibilities:
- Provide query/document embedding helpers with Qwen3 prompt emulation
  (prefixes for "query:" and "passage:"), matching the model card guidance.
- Power vector indexing and retrieval in `ingest.py` and `rag.py`.

Clinical focus: used throughout the autoimmune liver (AIH/PBC/PSC) RAG app.

Environment:
- `LMSTUDIO_BASE` is fixed here to a local server; adjust if needed.
- `EMBED_MODEL` can be overridden via environment.

Used by:
- `ingest.build_index` for document embeddings
- `rag.search_diverse` via `embed_queries`
- `rag.mmr` via `embed_texts`
"""

from openai import OpenAI
import numpy as np
import os
from typing import Optional, List, Tuple

LMSTUDIO_BASE = "http://192.168.1.2:1234/v1"
# Set this to the EXACT id shown by /v1/models in LM Studio
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-qwen3-embedding-0.6b")

client = OpenAI(base_url=LMSTUDIO_BASE, api_key="lm-studio")

# --- Qwen3 prompt emulation (see model card: use prompt_name="query" for queries) ---
QUERY_PREFIX = "query: "
DOC_PREFIX   = "passage: "   # optional; docs can also be sent raw

def prep_query(text: str) -> str:
    return QUERY_PREFIX + (text or "")

def prep_doc(text: str, title: Optional[str] = None) -> str:
    # You can include title if you like; Qwen3 doesn’t require it, but helps sometimes.
    # We’ll just prefix as a passage. If you want titles, uncomment the next line.
    # return DOC_PREFIX + (f"title: {title} | text: {text}" if title else text or "")
    return DOC_PREFIX + (text or "")

def debug_list_models() -> None:
    try:
        ms = client.models.list()
        print("LM Studio models available:")
        for m in ms.data:
            print(" -", m.id)
        print(f"Using EMBED_MODEL={EMBED_MODEL}")
    except Exception as e:
        print("Could not list models:", e)

def _embed_raw(texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.asarray(vecs, dtype="float32")

# Generic (used by MMR etc.)
def embed_texts(texts: List[str]) -> np.ndarray:
    return _embed_raw(texts)

# Query/doc helpers (use these in search + indexing)
def embed_queries(queries: List[str]) -> np.ndarray:
    return _embed_raw([prep_query(q) for q in queries])

def embed_docs(titled_chunks: List[Tuple[Optional[str], str]]) -> np.ndarray:
    """Embed document chunks.

    Expects a list of (title, text) pairs. Title may be None and is optional
    for Qwen3 embeddings; including it can sometimes help retrieval.
    """
    return _embed_raw([prep_doc(text, title) for title, text in titled_chunks])
