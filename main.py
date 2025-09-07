import argparse, glob, json, os, pickle, time, hashlib
import faiss
from ingest import build_index
from embedder_lms import debug_list_models, EMBED_MODEL
from llm_lms import generate_answer
from rag import search_diverse, make_prompt, mmr
from embedder_lms import debug_list_models, EMBED_MODEL, embed_queries, embed_texts

CACHE_DIR = ".cache"
INDEX_PATH = os.path.join(CACHE_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks.pkl")
META_PATH = os.path.join(CACHE_DIR, "meta.pkl")
MANIFEST_PATH = os.path.join(CACHE_DIR, "manifest.json")

def scan_pdfs(folder):
    return sorted(glob.glob(os.path.join(folder, "*.pdf")))

def file_fingerprint(path):
    st = os.stat(path)
    return {"path": os.path.abspath(path), "mtime": st.st_mtime_ns, "size": st.st_size}

def compute_manifest(pdf_paths, chunk_size, overlap):
    return {
        "timestamp": time.time(),
        "embed_model": EMBED_MODEL,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "files": [file_fingerprint(p) for p in pdf_paths],
        "digest": None,  # filled below
    }

def digest_manifest(m):
    h = hashlib.sha256()
    h.update(m["embed_model"].encode())
    h.update(str(m["chunk_size"]).encode())
    h.update(str(m["overlap"]).encode())
    for f in m["files"]:
        h.update(f["path"].encode())
        h.update(str(f["mtime"]).encode())
        h.update(str(f["size"]).encode())
    return h.hexdigest()

def load_cached():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH) and os.path.exists(META_PATH) and os.path.exists(MANIFEST_PATH)):
        return None
    try:
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f: chunks = pickle.load(f)
        with open(META_PATH, "rb") as f: meta = pickle.load(f)
        with open(MANIFEST_PATH, "r") as f: manifest = json.load(f)
        return {"index": index, "chunks": chunks, "meta": meta, "manifest": manifest}
    except Exception:
        return None

def save_cache(index, chunks, meta, manifest):
    os.makedirs(CACHE_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f: pickle.dump(chunks, f)
    with open(META_PATH, "wb") as f: pickle.dump(meta, f)
    with open(MANIFEST_PATH, "w") as f: json.dump(manifest, f, indent=2)

def needs_rebuild(new_manifest, existing_manifest):
    return (existing_manifest is None) or (existing_manifest.get("digest") != new_manifest.get("digest"))

def _label(m):
    if isinstance(m, dict):
        t = m.get("title"); p = m.get("page")
        if t and p: return f"{t} (p.{p})"
        if t: return t
    return str(m)

from rag import search_diverse, make_prompt, mmr
from embedder_lms import embed_queries, embed_texts  # embed_texts used inside MMR

def run_query(
    q,
    index,
    chunks,
    meta,
    k=10,
    fetch_k=80,
    per_file=2,
    use_mmr=True,
    mmr_lambda=0.7,
    threshold=0.25,   # NEW knob
):
    # 1) recall + diversify
    q_vec, picks = search_diverse(q, index, embed_queries, meta, fetch_k=fetch_k, per_file=per_file)

    # --- ADD THIS BLOCK HERE ---
    scores = [s for s, _ in picks]
    if not scores or max(scores) < threshold:
        print("\nQ:", q)
        print("\nA: I don't know based on the provided documents.")
        print("\nSources: (none above threshold)")
        return
    # ---------------------------

    cand_idxs = [i for _, i in picks]

    # 2) rerank with MMR (if enabled)
    if use_mmr:
        idxs = mmr(q_vec, cand_idxs, chunks, embed_texts, topn=k, lambda_mult=mmr_lambda)
    else:
        idxs = cand_idxs[:k]

    # 3) build prompt + ask LLM
    def _label(m):
        if isinstance(m, dict):
            t = m.get("title"); p = m.get("page")
            return f"{t} (p.{p})" if t and p else (t or str(m))
        return str(m)

    contexts = [(_label(meta[i]), chunks[i]) for i in idxs if i != -1]
    prompt = make_prompt(q, contexts)
    ans = generate_answer(prompt)

    print("\nQ:", q)
    print("\nA:", ans)
    print("\nSources:")
    for lbl, _ in contexts:
        print(" -", lbl)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="pdfs")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--chunk_size", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--ask_once", default="")  # optional one-shot question
    args = ap.parse_args()

    print(f"Scanning {args.folder}, found {len(scan_pdfs(args.folder))} PDFs")
    debug_list_models()

    pdfs = scan_pdfs(args.folder)
    new_manifest = compute_manifest(pdfs, args.chunk_size, args.overlap)
    new_manifest["digest"] = digest_manifest(new_manifest)

    cached = None if args.rebuild else load_cached()
    if cached and not needs_rebuild(new_manifest, cached.get("manifest")):
        print("Loaded cached index.")
        index, chunks, meta = cached["index"], cached["chunks"], cached["meta"]
    else:
        print("Building index (this computes embeddings once)...")
        index, chunks, meta = build_index(pdfs, chunk_size=args.chunk_size, overlap=args.overlap)
        save_cache(index, chunks, meta, new_manifest)
        print("Index cached to ./.cache")

    if args.ask_once:
        run_query(args.ask_once, index, chunks, meta, k=args.k)
        return

    # Interactive loop
    print("\nType your query (or just press Enter to exit):")
    while True:
        try:
            q = input("ask> ").strip()
        except EOFError:
            break
        if not q:
            break
        run_query(q, index, chunks, meta, k=args.k)

if __name__ == "__main__":
    main()