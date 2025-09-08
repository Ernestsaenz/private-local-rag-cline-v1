# embedder_lms.py
from openai import OpenAI
import numpy as np
import os

LMSTUDIO_BASE = "http://localhost:1234/v1"
# Set this to the EXACT id shown by /v1/models in LM Studio
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-qwen3-embedding-0.6b")

client = OpenAI(base_url=LMSTUDIO_BASE, api_key="lm-studio")

# --- Qwen3 prompt emulation (see model card: use prompt_name="query" for queries) ---
QUERY_PREFIX = "query: "
DOC_PREFIX   = "passage: "   # optional; docs can also be sent raw

def prep_query(text: str) -> str:
    return QUERY_PREFIX + (text or "")

def prep_doc(text: str, title: str | None = None) -> str:
    # You can include title if you like; Qwen3 doesn’t require it, but helps sometimes.
    # We’ll just prefix as a passage. If you want titles, uncomment the next line.
    # return DOC_PREFIX + (f"title: {title} | text: {text}" if title else text or "")
    return DOC_PREFIX + (text or "")

def debug_list_models():
    try:
        ms = client.models.list()
        print("LM Studio models available:")
        for m in ms.data:
            print(" -", m.id)
        print(f"Using EMBED_MODEL={EMBED_MODEL}")
    except Exception as e:
        print("Could not list models:", e)

def _embed_raw(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.asarray(vecs, dtype="float32")

# Generic (used by MMR etc.)
def embed_texts(texts: list[str]) -> np.ndarray:
    return _embed_raw(texts)

# Query/doc helpers (use these in search + indexing)
def embed_queries(queries: list[str]) -> np.ndarray:
    return _embed_raw([prep_query(q) for q in queries])

def embed_docs(titled_chunks: list[tuple[str | None, str]]) -> np.ndarray:
    # titled_chunks: [(title, text), ...]
    return _embed_raw([prep_doc(text, title) for title, text in titled_chunks])