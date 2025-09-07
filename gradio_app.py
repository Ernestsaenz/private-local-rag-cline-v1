import os, shutil, json, pickle, time, glob, hashlib
import gradio as gr
import faiss

# --- your modules ---
from ingest import build_index, load_pdfs  # you already have these
from rag import search_diverse, make_prompt, mmr
from llm_lms import generate_answer
from embedder_lms import debug_list_models, EMBED_MODEL, embed_queries, embed_texts

# ---- cache paths (same as main.py) ----
CACHE_DIR = ".cache"
INDEX_PATH = os.path.join(CACHE_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks.pkl")
META_PATH = os.path.join(CACHE_DIR, "meta.pkl")
MANIFEST_PATH = os.path.join(CACHE_DIR, "manifest.json")

PROMPTS_VERSION = "qwen3-v1"  # bump if you change prompt templates

# ---- globals ----
G_INDEX = None
G_CHUNKS = None
G_META = None
G_MANIFEST = None

def scan_pdfs(folder):
    return sorted(glob.glob(os.path.join(folder, "*.pdf")))

def file_fingerprint(path):
    st = os.stat(path)
    return {"path": os.path.abspath(path), "mtime": st.st_mtime_ns, "size": st.st_size}

def compute_manifest(pdf_paths, chunk_size, overlap):
    return {
        "timestamp": time.time(),
        "embed_model": EMBED_MODEL,
        "prompts_version": PROMPTS_VERSION,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "files": [file_fingerprint(p) for p in pdf_paths],
        "digest": None,
    }

def digest_manifest(m):
    h = hashlib.sha256()
    h.update(m["embed_model"].encode())
    h.update(m["prompts_version"].encode())
    h.update(str(m["chunk_size"]).encode())
    h.update(str(m["overlap"]).encode())
    for f in m["files"]:
        h.update(f["path"].encode()); h.update(str(f["mtime"]).encode()); h.update(str(f["size"]).encode())
    return h.hexdigest()

def load_cached():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH) and os.path.exists(META_PATH) and os.path.exists(MANIFEST_PATH)):
        return None
    try:
        ix = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f: ch = pickle.load(f)
        with open(META_PATH, "rb") as f: mt = pickle.load(f)
        with open(MANIFEST_PATH, "r") as f: mf = json.load(f)
        return {"index": ix, "chunks": ch, "meta": mt, "manifest": mf}
    except Exception:
        return None

def save_cache(index, chunks, meta, manifest):
    os.makedirs(CACHE_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f: pickle.dump(chunks, f)
    with open(META_PATH, "wb") as f: pickle.dump(meta, f)
    with open(MANIFEST_PATH, "w") as f: json.dump(manifest, f, indent=2)

def ensure_index(folder, chunk_size, overlap, force_rebuild=False):
    global G_INDEX, G_CHUNKS, G_META, G_MANIFEST
    pdfs = scan_pdfs(folder)
    new_m = compute_manifest(pdfs, chunk_size, overlap)
    new_m["digest"] = digest_manifest(new_m)

    if not force_rebuild:
        cached = load_cached()
        if cached and cached["manifest"].get("digest") == new_m["digest"]:
            G_INDEX, G_CHUNKS, G_META, G_MANIFEST = cached["index"], cached["chunks"], cached["meta"], cached["manifest"]
            return f"Loaded cached index. PDFs: {len(pdfs)}"
    # build
    G_INDEX, G_CHUNKS, G_META = build_index(pdfs, chunk_size=chunk_size, overlap=overlap)
    G_MANIFEST = new_m
    save_cache(G_INDEX, G_CHUNKS, G_META, G_MANIFEST)
    return f"Indexed {len(pdfs)} PDFs and cached."

def _label(m):
    if isinstance(m, dict):
        t = m.get("title"); p = m.get("page")
        return f"{t} (p.{p})" if t and p else (t or str(m))
    return str(m)

def ask(query, k, fetch_k, per_file, use_mmr, mmr_lambda, threshold):
    if not G_INDEX:
        return "Index not ready. Click Build/Load Index first.", ""
    # 1) recall + diversify
    q_vec, picks = search_diverse(query, G_INDEX, embed_queries, G_META, fetch_k=fetch_k, per_file=per_file)

    scores = [s for s, _ in picks]
    if not scores or max(scores) < threshold:
        return "I don't know based on the provided documents.", "Sources: (none above threshold)"

    cand_idxs = [i for _, i in picks]
    # 2) MMR or top-k
    if use_mmr:
        idxs = mmr(q_vec, cand_idxs, G_CHUNKS, embed_texts, topn=k, lambda_mult=mmr_lambda)
    else:
        idxs = cand_idxs[:k]
    contexts = [(_label(G_META[i]), G_CHUNKS[i]) for i in idxs if i != -1]
    prompt = make_prompt(query, contexts)
    ans = generate_answer(prompt)
    srcs = "Sources:\n" + "\n".join(f" - {lbl}" for lbl, _ in contexts)
    return ans, srcs

def list_pdfs(folder):
    pdfs = scan_pdfs(folder)
    return "\n".join(pdfs) if pdfs else "(no PDFs found)"

def add_uploads_to_folder(files, folder):
    if not files: 
        return "No files uploaded."
    os.makedirs(folder, exist_ok=True)
    added = []
    for f in files:
        dest = os.path.join(folder, os.path.basename(f.name))
        shutil.copyfile(f.name, dest)
        added.append(dest)
    return f"Saved: {len(added)} file(s) to {folder}"

# ---------- Gradio UI ----------
with gr.Blocks(title="Local RAG (Qwen3-Embedding + LM Studio)") as demo:
    gr.Markdown("## Local RAG — Qwen3-Embedding (LM Studio)\nMake sure LM Studio is running and serving your embedding + chat models.")

    with gr.Row():
        folder_in = gr.Textbox(value="pdfs", label="PDF folder path")
        btn_list = gr.Button("List PDFs")
    pdf_list = gr.Textbox(label="Found PDFs", lines=6)

    with gr.Row():
        upload = gr.File(label="Upload PDFs", file_count="multiple", file_types=[".pdf"])
        btn_save = gr.Button("Save uploads to folder")

    with gr.Row():
        chunk_size = gr.Slider(200, 1200, 500, step=50, label="Chunk size")
        overlap = gr.Slider(0, 400, 100, step=20, label="Overlap")
        btn_build = gr.Button("Build/Load Index")
    build_status = gr.Textbox(label="Index status")

    gr.Markdown("### Ask")
    with gr.Row():
        query = gr.Textbox(label="Query", placeholder="e.g., who presided each meeting?", lines=2)
    with gr.Row():
        k = gr.Slider(1, 8, 3, step=1, label="k (final contexts)")
        fetch_k = gr.Slider(10, 200, 80, step=10, label="FAISS candidates (fetch_k)")
        per_file = gr.Slider(1, 5, 2, step=1, label="Max per file (diversify)")
    with gr.Row():
        use_mmr = gr.Checkbox(value=True, label="Use MMR")
        mmr_lambda = gr.Slider(0.5, 0.95, 0.7, step=0.05, label="MMR lambda (relevance vs diversity)")
        threshold = gr.Slider(0.0, 0.9, 0.25, step=0.05, label="Similarity threshold (abstain below)")

    btn_ask = gr.Button("Ask")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Sources", lines=6)

    # wiring
    btn_list.click(list_pdfs, inputs=folder_in, outputs=pdf_list)
    btn_save.click(add_uploads_to_folder, inputs=[upload, folder_in], outputs=build_status)
    btn_build.click(ensure_index, inputs=[folder_in, chunk_size, overlap], outputs=build_status)
    btn_ask.click(ask, inputs=[query, k, fetch_k, per_file, use_mmr, mmr_lambda, threshold], outputs=[answer, sources])

if __name__ == "__main__":
    debug_list_models()
    demo.launch()