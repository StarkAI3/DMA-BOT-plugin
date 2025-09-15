import os
import uuid
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Local import of advanced RAG system
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from advanced_query_rag import AdvancedRAGQuerySystem


# Dynamic base directory - works in any environment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")


# Global RAG system instance
rag_system = None

# In-memory session store (in production, use Redis or database)
conversation_sessions: Dict[str, Dict[str, Any]] = {}

class ChatMessage(BaseModel):
    query: str
    session_id: Optional[str] = None
    
class ConversationHistory:
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.last_activity = datetime.now()
        
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
        self.last_activity = datetime.now()
        
        # Keep only last 10 messages to manage memory
        if len(self.messages) > 10:
            self.messages = self.messages[-10:]
    
    def get_recent_context(self, max_messages: int = 6) -> str:
        """Get recent conversation context as formatted string"""
        if not self.messages:
            return ""
            
        recent_messages = self.messages[-max_messages:]
        context_parts = []
        
        for msg in recent_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")
        
        return "\n".join(context_parts)

def clean_response_format(response_text: str) -> str:
    """Clean response text to remove JSON formatting and code blocks"""
    try:
        import json
        
        # Remove markdown code blocks (```json, ```, etc.)
        response_text = re.sub(r'```[\w]*\n?', '', response_text)
        response_text = re.sub(r'```', '', response_text)
        
        # Try to extract content from JSON structure if present
        if response_text.strip().startswith('{') and response_text.strip().endswith('}'):
            try:
                parsed = json.loads(response_text)
                if isinstance(parsed, dict) and 'answer' in parsed:
                    return parsed['answer']
            except:
                pass
        
        # Remove any remaining JSON-like patterns
        response_text = re.sub(r'^\s*{\s*"?\w+"?\s*:\s*"?', '', response_text)
        response_text = re.sub(r'"\s*}\s*$', '', response_text)
        
        # Clean up extra whitespace
        response_text = re.sub(r'\n+', ' ', response_text)
        response_text = re.sub(r'\s+', ' ', response_text)
        
        return response_text.strip()
        
    except Exception as e:
        print(f"Warning: Error cleaning response format: {e}")
        return response_text

def get_or_create_session(session_id: Optional[str] = None) -> tuple[str, ConversationHistory]:
    """Get existing session or create new one"""
    global conversation_sessions
    
    # Clean up old sessions (older than 1 hour)
    current_time = datetime.now()
    expired_sessions = [
        sid for sid, session in conversation_sessions.items()
        if current_time - session.last_activity > timedelta(hours=1)
    ]
    for sid in expired_sessions:
        del conversation_sessions[sid]
    
    if session_id and session_id in conversation_sessions:
        return session_id, conversation_sessions[session_id]
    else:
        # Create new session
        new_session_id = str(uuid.uuid4())
        conversation_sessions[new_session_id] = ConversationHistory()
        return new_session_id, conversation_sessions[new_session_id]

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
def api_chat(message: ChatMessage) -> Dict[str, Any]:
    global rag_system
    
    query = message.query
    if not query or not isinstance(query, str):
        raise HTTPException(status_code=400, detail="Field 'query' is required and must be a string.")

    # Get or create conversation session
    session_id, conversation = get_or_create_session(message.session_id)
    
    # Lightweight language detection: Devanagari range
    detected_lang = "mr" if any("\u0900" <= ch <= "\u097F" for ch in query) else "en"

    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")

    try:
        # Add user message to conversation history
        conversation.add_message("user", query)
        
        # Get conversation context for better understanding
        conversation_context = conversation.get_recent_context()
        
        # Minimal latency logging
        import time
        start = time.time()
        
        # Pass conversation context to RAG system
        result = rag_system.query(query, conversation_context=conversation_context)
        elapsed_ms = int((time.time() - start) * 1000)
        
        # Add assistant response to conversation history
        response_text = result.get("response", "")
        
        # Clean response format to remove JSON formatting issues
        response_text = clean_response_format(response_text)
        
        conversation.add_message("assistant", response_text, {"sources": result.get("sources", [])})
        
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
            "answer": response_text,
            "sources": norm_sources,
            "detected_lang": detected_lang,
            "latency_ms": elapsed_ms,
            "session_id": session_id,
            "conversation_length": len(conversation.messages),
            "metadata": result.get("metadata", {})
        }
    except Exception as exc:
        # Surface a safe error to client
        raise HTTPException(status_code=500, detail=str(exc))


# Optional: health check
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


