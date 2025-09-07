from pypdf import PdfReader
import faiss, numpy as np
from embedder_lms import embed_texts

def load_pdfs(paths):
    docs = []
    for p in paths:
        txts = []
        reader = PdfReader(p)
        for pg in reader.pages:
            t = (pg.extract_text() or "").strip()
            if t: txts.append(t)
        full = "\n".join(txts)
        if full:
            docs.append({"title": p, "text": full})
    return docs

def chunk(text, size=800, overlap=120):
    words, out, step = text.split(), [], max(1, size - overlap)
    for i in range(0, len(words), step):
        part = " ".join(words[i:i+size])
        if part.strip(): out.append(part)
    return out

def build_index(pdf_paths):
    docs = load_pdfs(pdf_paths)
    chunks, meta = [], []
    for d in docs:
        for c in chunk(d["text"]):
            chunks.append(c); meta.append(d["title"])
    X = embed_texts(chunks)
    faiss.normalize_L2(X)                           # cosine via inner product
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    return index, chunks, meta