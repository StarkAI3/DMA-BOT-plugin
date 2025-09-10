import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Local import of advanced RAG system
from .advanced_query_rag import AdvancedRAGQuerySystem


BASE_DIR = "/home/stark/Desktop/DMA_BOT"
STATIC_DIR = os.path.join(BASE_DIR, "static")


# Global RAG system instance
rag_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global rag_system
    try:
        # Initialize the advanced RAG system
        rag_system = AdvancedRAGQuerySystem()
        rag_system.initialize()
        print("✅ Advanced RAG system initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        # Do not crash startup; rely on lazy load fallback
        pass
    
    yield
    
    # Shutdown (if needed)
    pass


app = FastAPI(title="DMA Assistant", version="1.0.0", lifespan=lifespan)

# CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_index() -> FileResponse:
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


# Serve static assets at /static
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.post("/api/chat")
def api_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    global rag_system
    
    query: Optional[str] = (payload or {}).get("query")
    if not query or not isinstance(query, str):
        raise HTTPException(status_code=400, detail="Field 'query' is required and must be a string.")

    # Lightweight language detection: Devanagari range
    detected_lang = "mr" if any("\u0900" <= ch <= "\u097F" for ch in query) else "en"

    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")

    try:
        # Minimal latency logging
        import time
        start = time.time()
        result = rag_system.query(query)
        elapsed_ms = int((time.time() - start) * 1000)
        
        # Normalize sources for the frontend (array of {url, title})
        norm_sources = []
        for source in result.get("sources", []):
            if not source:
                continue
            norm_sources.append({
                "url": source.get("url", ""),
                "title": source.get("title", "Unknown"),
                "category": source.get("category", "general"),
                "relevance_score": source.get("relevance_score", 0.0)
            })

        return {
            "answer": result.get("response", ""),
            "sources": norm_sources,
            "detected_lang": detected_lang,
            "latency_ms": elapsed_ms,
            "metadata": result.get("metadata", {})
        }
    except Exception as exc:
        # Surface a safe error to client
        raise HTTPException(status_code=500, detail=str(exc))


# Optional: health check
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


