"""
PDF ingestion and FAISS index construction for local clinical RAG.

Typical input PDFs: AIH, PBC, PSC guidelines/consensus or key reviews.

Workflow:
- `load_pdfs` extracts text per page for each PDF.
- `chunk_page` splits pages into overlapping word windows.
- `build_index` embeds all chunks and builds a normalized inner-product FAISS index.

Outputs:
- `index`: FAISS `IndexFlatIP` (on L2-normalized vectors)
- `chunks`: list[str] of chunk texts aligned with index ids
- `meta`: list[dict] with `{"title": <path>, "page": <int>}` aligned with chunks

Used by `main.py` and `gradio_app.py` for retrieval.
"""

from pypdf import PdfReader
import os
import faiss
from typing import List, Tuple, Dict
from embedder_lms import embed_docs

def load_pdfs(paths: List[str]) -> List[Dict]:
    """Load PDFs and extract text per page.

    Returns a list of dicts: {"title": path, "pages": [{"page": 1, "text": ...}, ...]}
    Pages without extractable text are skipped.
    """
    docs: List[Dict] = []
    for p in paths:
        reader = PdfReader(p)
        pages: List[Dict] = []
        for i, pg in enumerate(reader.pages):
            txt = (pg.extract_text() or "").strip()
            if txt:
                pages.append({"page": i+1, "text": txt})
        if pages:
            docs.append({"title": p, "pages": pages})
    return docs

def chunk_page(text: str, size: int = 500, overlap: int = 100) -> List[str]:
    """Split page text into overlapping word-based chunks.

    `size` is the max words per chunk. `overlap` words are carried into the
    next window to preserve context.
    """
    words = text.split()
    step = max(1, size - overlap)
    return [" ".join(words[i:i+size]) for i in range(0, len(words), step)]

def build_index(pdf_paths: List[str], chunk_size: int = 500, overlap: int = 100):
    """Build embeddings and FAISS index for a set of PDFs."""
    docs = load_pdfs(pdf_paths)
    chunks: List[str] = []
    meta: List[Dict] = []
    titled_chunks: List[Tuple[str, str]] = []
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
