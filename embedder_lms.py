# embedder_lms.py
from openai import OpenAI
import numpy as np, sys

LMSTUDIO_BASE = "http://localhost:1234/v1"
EMBED_MODEL = "text-embedding-embeddinggemma-300m"  # replace with EXACT id from /v1/models
client = OpenAI(base_url=LMSTUDIO_BASE, api_key="lm-studio")

def debug_list_models():
    try:
        ms = client.models.list()
        print("LM Studio models available:")
        for m in ms.data:
            print(" -", m.id)
        print(f"Using EMBED_MODEL={EMBED_MODEL}")
    except Exception as e:
        print("Could not list models:", e)

def embed_texts(texts):
    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        vecs = [d.embedding for d in resp.data]
        return np.asarray(vecs, dtype="float32")
    except Exception as e:
        # Give a clearer hint if the wrong model type is loaded
        msg = str(e)
        if "No models loaded" in msg or "model_not_found" in msg:
            raise RuntimeError(
                f"Embeddings model '{EMBED_MODEL}' not found on LM Studio.\n"
                "Open LM Studio → Developer → Local Server and load an **Embeddings** model "
                "(e.g., google/embedding-gemma-300m). Then set EMBED_MODEL to that exact id "
                "as shown by /v1/models."
            ) from e
        if "Model is not embedding" in msg:
            raise RuntimeError(
                f"Model '{EMBED_MODEL}' is not an embeddings model.\n"
                "In LM Studio, load the Embeddings variant (task: Embeddings), not a chat/qat build."
            ) from e
        raise