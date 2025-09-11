# DMA Assistant (RAG System) – Directorate of Municipal Administration, Maharashtra

A production-ready Retrieval Augmented Generation (RAG) system that powers the DMA Assistant for citizens and staff of the Directorate of Municipal Administration, Maharashtra. It ingests official web content, tabular datasets, and services catalogs, builds high-quality semantic chunks, indexes them into Pinecone using `intfloat/e5-base-v2` embeddings, and serves fast, accurate answers using Google Gemini 2.0 Flash with a responsive web UI.

---

## Highlights
- Accurate municipal information from curated sources (web pages, tabular data, services)
- Robust semantic chunking and metadata enrichment
- Vector search on Pinecone with intent-aware filtering and reranking
- Response generation via Gemini 2.0 Flash
- Multilingual query support (English and Marathi)
- FastAPI backend with sessioned conversations
- Modern, responsive web interface

---

## Repository Structure
```
/home/stark/Desktop/DMA_BOT
├── env.sample                    # Template for environment variables
├── requirements.txt              # Python dependencies
├── SETUP_GUIDE.md                # Step-by-step setup/run guide
├── quick_setup.py                # Environment & model verification
├── rebuild_rag_system.py         # Orchestrates full rebuild (preprocess → embed → frontend/server)
├── src/
│   ├── server.py                 # FastAPI app serving API and static UI
│   ├── advanced_preprocess.py    # Advanced semantic chunking pipeline
│   ├── advanced_embed_upsert.py  # Embedding and Pinecone upsert pipeline
│   ├── advanced_query_rag.py     # RAG query engine (retrieval + Gemini)
│   ├── scraper_v1.py             # Web scraper for DMA content
│   └── tabular_scraper.py        # Scraper/ingestor for tabular datasets
├── static/                       # Web UI (index.html, main.js, styles.css, seal.svg)
├── final_data/                   # Curated source JSONs (web, tabular, services)
├── optimized_data/               # Outputs: chunks, manifest, embedding report
├── models/                       # Local model cache (Hugging Face)
├── robust_queries.txt            # Ready-to-run test queries (EN/MR)
└── venv/                         # Project virtual environment (optional)
```

---

## System Architecture
- Data ingestion: `scraper_v1.py`, `tabular_scraper.py` build curated JSON from `mahadma.maharashtra.gov.in` and related sources
- Preprocessing: `advanced_preprocess.py` performs semantic-aware chunking with:
  - sentence-boundary segmentation
  - entity/keyword extraction
  - service category tagging
  - importance scoring and metadata enrichment
- Embeddings & Indexing: `advanced_embed_upsert.py` encodes chunks using `intfloat/e5-base-v2` and upserts to Pinecone index `dma-knowledge-base-v3`
- Query Engine: `advanced_query_rag.py`
  - query preprocessing and intent detection
  - metadata-driven filtered vector search
  - reranking to select best context
  - Gemini 2.0 Flash generation with conversation awareness
- Backend API: `src/server.py` (FastAPI) exposes `/api/chat`, `/health` and serves static UI
- Frontend: `static/index.html`, `static/main.js`, `static/styles.css` – clean, responsive chat with typing indicator and session management

---

## Tech Stack
- Language: Python 3.11
- Web: FastAPI, CORS middleware, StaticFiles
- LLM: Google Gemini 2.0 Flash (`google-generativeai`)
- Embeddings: `intfloat/e5-base-v2` (SentenceTransformers)
- Vector DB: Pinecone (serverless, cosine metric)
- Scraping: requests, BeautifulSoup, pandas, lxml
- NLP Utilities: nltk
- Tooling: tqdm, python-dotenv, pydantic

---

## Performance & Quality
From the last successful build:
- optimized_data/manifest.json
  - total_chunks: 882
  - high_quality_percentage: 84.35%
- optimized_data/embedding_report.json
  - model_used: e5-base-v2 (dim 768)
  - total_vectors: 882 (100% upsert success)
  - avg_embedding_time: ~212 ms / chunk

---

## Prerequisites
- Python 3.11
- Valid API keys:
  - Pinecone: PINECONE_API_KEY
  - Google Generative AI: GEMINI_API_KEY
- Internet access for model downloads and Pinecone

Optional but recommended:
- Linux environment (tested on Ubuntu 5.15 kernel)
- Virtual environment (`venv`)

---

## Installation
1) Create and activate a virtual environment
```bash
cd /home/stark/Desktop/DMA_BOT
python3 -m venv venv
source venv/bin/activate
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Configure environment
- Copy `env.sample` to `.env` and fill values
```bash
cp env.sample .env
# edit .env (PINECONE_API_KEY, GEMINI_API_KEY, etc.)
```

4) Quick verification (optional)
```bash
python quick_setup.py
```

---

## Data Pipeline (ETL)
- Sources reside in `final_data/`:
  - `final_data/normal data/` – web content JSONs
  - `final_data/tabular data/` – tabular datasets
  - `final_data/services.json` – service catalog

1) Preprocess into semantic chunks
```bash
python src/advanced_preprocess.py
# Outputs → optimized_data/{web_content_chunks.jsonl, tabular_data_chunks.jsonl, service_chunks.jsonl, all_chunks.jsonl}
# Also writes optimized_data/manifest.json
```

2) Embed + upsert to Pinecone
```bash
python src/advanced_embed_upsert.py
# Uses intfloat/e5-base-v2, writes optimized_data/embedding_report.json
```

3) One-shot orchestrated rebuild (all steps + server/frontend refresh)
```bash
python rebuild_rag_system.py
```

---

## Running the App
- Start FastAPI backend and serve UI
```bash
python src/server.py
# Visit http://localhost:5000
```

- Health check
```bash
curl http://localhost:5000/health
```

---

## API
- POST `/api/chat`
  - Request (JSON):
    ```json
    { "query": "How to apply for trade license?", "session_id": "optional-session-uuid" }
    ```
  - Response (JSON):
    ```json
    {
      "answer": "...",
      "sources": [{"url":"...","title":"...","category":"...","relevance_score":0.93}],
      "detected_lang": "en|mr",
      "latency_ms": 1234,
      "session_id": "...",
      "conversation_length": 4,
      "metadata": { "results_found": 20, "context_chunks": 8, "intent": {"intents":["apply"],"service_categories":["trade"],"is_procedural":true,"needs_contact":false} }
    }
    ```

---

## Frontend
- Chat UI with:
  - welcome screen and start button
  - session handling and timestamps
  - typing indicator
  - link auto-formatting
  - Marathi/English language indicator per message

Files:
- `static/index.html`
- `static/main.js`
- `static/styles.css`

---

## Configuration (.env)
Key variables (see `env.sample` for full list):
- PINECONE_API_KEY, PINECONE_INDEX_NAME (default: dma-knowledge-base-v3)
- GEMINI_API_KEY, GEMINI_MODEL=gemini-2.0-flash
- EMBEDDING_MODEL=intfloat/e5-base-v2, EMBEDDING_DEVICE=cpu
- Retrieval: TOP_K_RETRIEVAL=20, FINAL_CONTEXT_LIMIT=8, MIN_SIMILARITY_THRESHOLD=0.3
- Server: SERVER_HOST=0.0.0.0, SERVER_PORT=5000

---

## Security & Privacy
- API keys are read from `.env` (do not commit `.env`)
- CORS is open by default; restrict `allow_origins` for production
- Sessions are in-memory; use Redis/DB for production, and configure `SECRET_KEY`
- HTTPS not enabled by default; terminate TLS at a reverse proxy or enable certs
- Scraping respects timeouts and delay configuration

---

## Testing & QA
- Manual CLI test
```bash
python src/advanced_query_rag.py
```
- Web test
  - Run server and browse to `http://localhost:5000`
  - Try queries from `robust_queries.txt` (EN/MR)
- Health check: `/health`

---

## Troubleshooting
- Model not loading:
  - Ensure internet access; `quick_setup.py` validates `intfloat/e5-base-v2`
- Missing API keys:
  - Export `PINECONE_API_KEY` and `GEMINI_API_KEY` or set in `.env`
- Pinecone issues:
  - Confirm index name/region; list indexes to verify connectivity
- Memory limits:
  - Lower `BATCH_SIZE` in `advanced_embed_upsert.py`
- Empty answers:
  - Verify chunks in `optimized_data/all_chunks.jsonl` and index stats; adjust `TOP_K_RETRIEVAL` / `FINAL_CONTEXT_LIMIT`

---

## Maintenance
- Rebuild end-to-end when data updates: `python rebuild_rag_system.py`
- Monitor vector counts and latency via logs and `embedding_report.json`
- Keep dependencies updated: `pip install -r requirements.txt --upgrade`

---

## Roadmap (suggested)
- Production session store (Redis) and auth for admin endpoints
- Query analytics dashboard and feedback loop
- Offline batch verification tests with expected answers
- Multilingual generation improvements and Marathi templates
- Dockerization and CI/CD

---

## License
Internal project for DMA Assistant. Licensing to be defined by the organization.

