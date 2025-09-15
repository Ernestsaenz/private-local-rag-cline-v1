# Private Local RAG for Autoimmune Liver Diseases (AIH, PBC, PSC)

A local, production‑ready Retrieval‑Augmented Generation (RAG) system focused on the clinical management of autoimmune liver diseases:
- Autoimmune Hepatitis (AIH)
- Primary Biliary Cholangitis (PBC)
- Primary Sclerosing Cholangitis (PSC)

Everything runs on your machine using LM Studio — no external APIs or cloud access.

## ✨ Features

- 🔍 **Semantic search across PDFs** using Qwen3 embeddings
- 📄 **Automatic PDF parsing and chunking** with configurable parameters
- 🗂 **Intelligent caching** (embeddings + index saved in `.cache/`)
- 🎛 **Configurable parameters**: chunk size, overlap, retrieval settings
- 💻 **Local inference** with LM Studio (no API calls, no cloud)
- 🌐 **Dual interface**: Command-line + Web UI (Gradio)
- 📊 **Source citations** with page references
- 🩺 **Clinical focus**: tuned instructions for AIH/PBC/PSC decision support
- 🧭 **Abstain behavior**: configurable similarity threshold for “I don’t know”

> Safety: This tool assists clinicians by surfacing guideline‑based information from your uploaded PDFs. It does not provide medical advice and should not replace clinical judgment or local protocols.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- LM Studio installed and running
- Models loaded in LM Studio

### Installation
```bash
# Clone and setup
git clone <your-repo>
cd private-local-rag

# Install dependencies
python3.11 -m pip install -r requirements.txt
```

### Usage

**Option 1: Easy Startup**
```bash
python3.11 start.py
```

**Option 2: Direct Commands**
```bash
# Web UI
python3.11 gradio_app.py

# Command Line
python3.11 main.py --ask_once "What is this document about?"
```

## 📁 Project Structure

```
private-local-rag/
├── main.py              # CLI for local clinical Q&A over PDFs
├── gradio_app.py        # Web UI (AIH/PBC/PSC focus)  
├── embedder_lms.py      # LM Studio embedding integration
├── llm_lms.py           # LM Studio chat integration (autoimmune liver)
├── ingest.py            # PDF processing and chunking
├── rag.py               # Retrieval + prompt assembly (MMR reranking)
├── start.py             # Easy startup script
├── requirements.txt     # Dependencies
├── pdfs/                # Your guideline/consensus PDFs (AIH/PBC/PSC)
└── .cache/              # Cached embeddings and index

## 🩺 Clinical Focus

This project targets autoimmune liver diseases commonly managed in hepatology:
- AIH: diagnosis (serology, IgG, histology), initial/maintenance therapy (steroids, azathioprine), monitoring and tapering, escalation options, special populations.
- PBC: diagnosis (ALP, AMA), first‑line ursodeoxycholic acid (UDCA), inadequate response algorithms (e.g., OCA/fibrates per your PDFs), pruritus management, transplant considerations.
- PSC: diagnosis (cholestatic pattern, cholangiography), dominant strictures, cholangitis management, IBD surveillance, dysplasia/CCA risk, transplant referral.

Answers come strictly from your uploaded PDFs; conflicting recommendations are presented separately with page‑level citations.
```

## ⚙️ Configuration

### LM Studio Setup
1. **Start LM Studio** and go to "Local Server" tab
2. **Load embedding model**: `text-embedding-qwen3-embedding-0.6b`
3. **Load chat model**: `qwen/qwen3-1.7b`
4. **Start server** on port 1234

### Environment Variables
```bash
# Optional: Override default models
export EMBED_MODEL="text-embedding-qwen3-embedding-0.6b"
export LLM_MODEL="qwen/qwen3-1.7b"
```

## 🎯 Usage Examples

### Command Line
```bash
# Interactive mode
python3.11 main.py

# One-shot question
python3.11 main.py --ask_once "Initial treatment for AIH flare?"

# Custom parameters
python3.11 main.py --chunk_size 800 --overlap 150 --k 5
```

### Web Interface
```bash
python3.11 gradio_app.py
# Open: http://127.0.0.1:7860
```

## 🔧 Advanced Configuration

### Retrieval Parameters
- `--chunk_size`: Text chunk size (200-1200, default: 500)
- `--overlap`: Chunk overlap (0-400, default: 100)  
- `--k`: Final contexts returned (default: 3)
- `--fetch_k`: FAISS candidates retrieved (default: 80)
- `--per_file`: Max chunks per document (default: 2)

### MMR Reranking
- `--use_mmr`: Enable MMR reranking (default: True)
- `--mmr_lambda`: Relevance vs diversity balance (0.5-0.95, default: 0.7)
- `--threshold`: Similarity threshold for "I don't know" (default: 0.25)

## 📊 Performance Notes

- **First run**: Slow (computing embeddings)
- **Subsequent runs**: Instant (cached index)
- **Model switching**: Rebuild index with `--rebuild`
- **Memory usage**: ~500MB for typical document set

## 🛠 Troubleshooting

### Common Issues
1. **"Connection refused"**: LM Studio server not running
2. **"No models found"**: Models not loaded in LM Studio
3. **"I don't know"**: Similarity threshold too high

### Debug Commands
```bash
# Test LM Studio connection
python3.11 -c "from embedder_lms import debug_list_models; debug_list_models()"

# Rebuild index
python3.11 main.py --rebuild
```

## 📈 Roadmap

- [ ] Add cross‑encoder reranker for sharper results
- [ ] Multi-modal document support (images, tables)
- [ ] Export answers + citations as JSON/Markdown
- [ ] Batch query processing
- [ ] Query history and analytics
 - [ ] Disease selector (AIH/PBC/PSC prompt nuances) in UI

## 📚 Your Documents

Place your guideline/consensus/review PDFs into `pdfs/`. Examples:
- AIH: international guidelines, society statements, key reviews.
- PBC: diagnostic and treatment guidelines, response criteria.
- PSC: diagnostic, surveillance, and management guidance.

## 🛡️ Safety & Privacy

- Runs fully offline on your machine. No network calls from inference.
- Intended for clinician use; not a substitute for medical advice.
- Always confirm recommendations against local protocols and the full source text.

## 📄 License

Prototype for experimentation. No warranty.

