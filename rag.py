import faiss

def search(query, index, embed_fn, k=5):
    import numpy as np
    q = embed_fn([query]).astype("float32")
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    return D[0], I[0]

def make_prompt(query, contexts):
    ctx = "\n\n".join([f"[{t}]\n{c}" for t, c in contexts])
    return f"""Answer the question only using the sources below.
If missing, say you don't know. Cite sources in parentheses by filename.

Question: {query}

Sources:
{ctx}

Answer:"""