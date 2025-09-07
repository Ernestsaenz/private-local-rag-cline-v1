# rag.py
import faiss, numpy as np
from collections import defaultdict

def search_diverse(query, index, embed_fn, meta, fetch_k=80, per_file=2):
    q = embed_fn([query]).astype("float32")
    faiss.normalize_L2(q)
    D, I = index.search(q, fetch_k)

    picks, seen = [], defaultdict(int)
    for d, i in zip(D[0], I[0]):
        if i == -1:
            continue
        fname = meta[i]["title"] if isinstance(meta[i], dict) else str(meta[i])
        if seen[fname] < per_file:
            picks.append((float(d), i))
            seen[fname] += 1
    return q, picks  # return q (query vec) for MMR

def mmr(query_vec, cand_idxs, chunks, embed_texts, topn=5, lambda_mult=0.7):
    """
    MMR with embedding-only signals.
    - query_vec: shape (1, D) L2-normalized
    - cand_idxs: list[int]
    """
    if not cand_idxs:
        return []
    # embed candidate chunks with the SAME embedder used for docs
    cand_vecs = embed_texts([chunks[i] for i in cand_idxs]).astype("float32")
    faiss.normalize_L2(cand_vecs)

    selected = []
    remaining = list(range(len(cand_idxs)))
    rel = (cand_vecs @ query_vec.T).ravel()  # relevance to query

    while remaining and len(selected) < topn:
        if not selected:
            j = int(np.argmax(rel[remaining]))
            selected.append(remaining.pop(j))
            continue
        # diversity: max sim to any already selected
        S = cand_vecs[selected]
        div = (cand_vecs[remaining] @ S.T).max(axis=1)
        score = lambda_mult * rel[remaining] - (1 - lambda_mult) * div
        j = int(np.argmax(score))
        selected.append(remaining.pop(j))

    return [cand_idxs[j] for j in selected]

def make_prompt(query, contexts, max_context_chars=9000, per_snippet_max=1500):
    """
    Build the LLM prompt from query + retrieved contexts.
    contexts: list of (label, text) tuples
    """
    chunks = []
    total = 0
    for label, text in contexts:
        snippet = text[:per_snippet_max]
        block = f"[{label}]\n{snippet}\n\n"
        if total + len(block) > max_context_chars:
            break
        chunks.append(block)
        total += len(block)

    ctx = "".join(chunks)
    return f"""Answer the question strictly based on the SOURCES below.
If different sources say different things, list them separately.
Do not merge them.

Question: {query}

SOURCES:
{ctx}
Answer:"""