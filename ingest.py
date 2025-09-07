# ingest.py
from pypdf import PdfReader
import os
import faiss
from embedder_lms import embed_docs

def load_pdfs(paths):
    docs = []
    for p in paths:
        reader = PdfReader(p)
        pages = []
        for i, pg in enumerate(reader.pages):
            txt = (pg.extract_text() or "").strip()
            if txt:
                pages.append({"page": i+1, "text": txt})
        if pages:
            docs.append({"title": p, "pages": pages})
    return docs

def chunk_page(text, size=500, overlap=100):
    words = text.split()
    step = max(1, size - overlap)
    return [" ".join(words[i:i+size]) for i in range(0, len(words), step)]

def build_index(pdf_paths, chunk_size=500, overlap=100):
    docs = load_pdfs(pdf_paths)
    chunks, meta, titled_chunks = [], [], []
    for d in docs:
        fname = os.path.basename(d["title"])
        for page in d["pages"]:
            for c in chunk_page(page["text"], size=chunk_size, overlap=overlap):
                chunks.append(c)
                meta.append({"title": d["title"], "page": page["page"]})
                titled_chunks.append((fname, c))  # include filename if you want title signal later

    X = embed_docs(titled_chunks)
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    return index, chunks, meta