from ingest import build_index
from embedder_lms import embed_texts
from rag import search, make_prompt
from llm_lms import generate_answer
import glob, os

# 1) collect all PDFs in a folder
FOLDER = "pdfs"   # relative or absolute path
PDFS = glob.glob(os.path.join(FOLDER, "*.pdf"))
print(f"Scanning {FOLDER}, found {len(PDFS)} PDFs")

# 2) build index
from embedder_lms import debug_list_models
debug_list_models()
index, chunks, meta = build_index(PDFS)

# 3) ask questions
def ask(q, k=5):
    _, idxs = search(q, index, embed_texts, k=k)
    contexts = [(meta[i], chunks[i]) for i in idxs if i != -1]
    prompt = make_prompt(q, contexts)
    ans = generate_answer(prompt)
    print(ans)
    print("\nSources:")
    for m, _ in contexts:
        print(" -", m)

# example
ask("when were the New York City Council meetings held in 2025?")
# ask("who's the speaker?")