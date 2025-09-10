#!/usr/bin/env python3
"""
Advanced RAG Query System with Gemini 2.0 Flash
Optimized for maximum accuracy and context-aware responses
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Vector database and embedding imports
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai

# Configuration
BASE_DIR = "/home/stark/Desktop/DMA_BOT"
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "optimized_data")

# Pinecone configuration
PINECONE_INDEX_NAME = "dma-knowledge-base-v3"

# Query optimization parameters
TOP_K_RETRIEVAL = 20  # Retrieve more candidates for reranking
FINAL_CONTEXT_LIMIT = 8  # Final number of chunks to use
MIN_SIMILARITY_THRESHOLD = 0.3  # Minimum similarity for relevance
CONTEXT_WINDOW_LIMIT = 6000  # Characters limit for Gemini context

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedRAGQuerySystem:
    """Advanced RAG system with intelligent retrieval and context optimization"""
    
    def __init__(self):
        self.embedding_model = None
        self.pc = None
        self.index = None
        self.model_name = None
        
        # Initialize Gemini [[memory:8625932]]
        self.gemini_model = None
        self.setup_gemini()
        
    def setup_embedding_model(self) -> bool:
        """Setup E5-base-v2 embedding model for query encoding"""
        try:
            self.embedding_model = SentenceTransformer('intfloat/e5-base-v2', device="cpu")
            self.model_name = "e5-base-v2"
            logger.info("Loaded E5-base-v2 embedding model (768-dim)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup embedding model: {e}")
            return False
    
    def setup_gemini(self) -> bool:
        """Setup Gemini 2.0 Flash model"""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                logger.error("GEMINI_API_KEY environment variable not set")
                return False
            
            genai.configure(api_key=api_key)
            
            # Use Gemini 2.0 Flash for optimal performance [[memory:8625932]]
            self.gemini_model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config={
                    "temperature": 0.1,  # Low temperature for accuracy
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                }
            )
            
            logger.info("Gemini 2.0 Flash model initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Gemini: {e}")
            return False
    
    def setup_pinecone(self) -> bool:
        """Setup Pinecone connection"""
        try:
            api_key = os.getenv('PINECONE_API_KEY')
            if not api_key:
                logger.error("PINECONE_API_KEY environment variable not set")
                return False
            
            self.pc = Pinecone(api_key=api_key)
            self.index = self.pc.Index(PINECONE_INDEX_NAME)
            
            # Test connection
            stats = self.index.describe_index_stats()
            logger.info(f"Connected to Pinecone. Vectors available: {stats.total_vector_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Pinecone: {e}")
            return False
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess and enhance user query"""
        # Clean the query
        query = query.strip()
        
        # Expand common abbreviations
        expansions = {
            "noc": "no objection certificate",
            "ulb": "urban local body",
            "dma": "directorate of municipal administration",
            "mc": "municipal council",
            "mp": "municipal corporation"
        }
        
        query_lower = query.lower()
        for abbr, expansion in expansions.items():
            if abbr in query_lower:
                query = re.sub(rf'\b{re.escape(abbr)}\b', expansion, query, flags=re.IGNORECASE)
        
        return query
    
    def extract_query_intent(self, query: str) -> Dict[str, Any]:
        """Extract intent and entities from query"""
        query_lower = query.lower()
        
        intent_keywords = {
            "apply": ["apply", "application", "form", "submit", "register"],
            "information": ["what", "how", "where", "when", "info", "details"],
            "contact": ["contact", "phone", "email", "address", "office"],
            "procedure": ["procedure", "process", "steps", "how to"],
            "status": ["status", "track", "check", "progress"],
            "fee": ["fee", "cost", "charges", "payment", "amount"],
            "documents": ["documents", "papers", "requirements", "needed"]
        }
        
        detected_intents = []
        for intent, keywords in intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intents.append(intent)
        
        # Extract service categories
        service_categories = {
            "water": ["water", "connection", "bill", "supply"],
            "property": ["property", "tax", "assessment"],
            "trade": ["trade", "license", "business"],
            "marriage": ["marriage", "registration", "certificate"],
            "birth": ["birth", "certificate"],
            "death": ["death", "certificate"],
            "noc": ["noc", "permission", "approval"]
        }
        
        detected_categories = []
        for category, keywords in service_categories.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_categories.append(category)
        
        return {
            "intents": detected_intents if detected_intents else ["information"],
            "service_categories": detected_categories,
            "is_procedural": any(word in query_lower for word in ["how", "procedure", "steps", "process"]),
            "needs_contact": any(word in query_lower for word in ["contact", "phone", "office", "address"])
        }
    
    def build_search_filters(self, query_intent: Dict[str, Any]) -> Dict[str, Any]:
        """Build metadata filters for targeted search"""
        filters = {}
        
        # Filter by service categories if detected
        if query_intent["service_categories"]:
            category_conditions = []
            for category in query_intent["service_categories"]:
                category_conditions.append({"content_category": {"$eq": f"{category}_services"}})
            
            if len(category_conditions) == 1:
                filters.update(category_conditions[0])
            else:
                filters["$or"] = category_conditions
        
        # Boost service-related content for application queries
        if "apply" in query_intent["intents"]:
            filters["service_related"] = {"$eq": "true"}
        
        # Filter for contact information
        if query_intent["needs_contact"]:
            filters["$or"] = [
                {"content_category": {"$eq": "organizational"}},
                {"chunk_type": {"$eq": "table_row"}},
                {"keywords": {"$in": ["contact", "phone", "address"]}}
            ]
        
        return filters
    
    def semantic_search(self, query: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform semantic search with filtering"""
        try:
            # Encode the query
            query_embedding = self.embedding_model.encode(query, normalize_embeddings=True).tolist()
            
            # Perform vector search
            search_kwargs = {
                "vector": query_embedding,
                "top_k": TOP_K_RETRIEVAL,
                "include_metadata": True
            }
            
            if filters:
                search_kwargs["filter"] = filters
            
            results = self.index.query(**search_kwargs)
            
            # Extract and format results
            formatted_results = []
            for match in results.matches:
                if match.score >= MIN_SIMILARITY_THRESHOLD:
                    formatted_results.append({
                        "id": match.id,
                        "score": match.score,
                        "text": match.metadata.get("text", ""),
                        "metadata": match.metadata
                    })
            
            logger.info(f"Retrieved {len(formatted_results)} relevant chunks (score >= {MIN_SIMILARITY_THRESHOLD})")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def rerank_results(self, query: str, results: List[Dict[str, Any]], query_intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rerank results based on query intent and content quality"""
        
        def calculate_relevance_score(result: Dict[str, Any]) -> float:
            base_score = result["score"]
            metadata = result["metadata"]
            text = result["text"].lower()
            query_lower = query.lower()
            
            # Boost based on importance score
            importance_boost = float(metadata.get("importance_score", 0.5)) * 0.2
            
            # Boost for exact keyword matches
            query_words = set(query_lower.split())
            text_words = set(text.split())
            keyword_overlap = len(query_words.intersection(text_words)) / len(query_words)
            keyword_boost = keyword_overlap * 0.3
            
            # Boost based on content type preferences
            content_type_boost = 0.0
            content_category = metadata.get("content_category", "")
            
            if "apply" in query_intent["intents"] and "service" in content_category:
                content_type_boost += 0.3
            
            if query_intent["needs_contact"] and content_category == "organizational":
                content_type_boost += 0.4
            
            if query_intent["is_procedural"] and "procedure" in text:
                content_type_boost += 0.2
            
            # Boost for service-related content
            if metadata.get("service_related") == "true":
                content_type_boost += 0.1
            
            # Penalize very short or very long chunks
            text_length = len(result["text"])
            length_penalty = 0.0
            if text_length < 100:
                length_penalty = -0.2
            elif text_length > 1000:
                length_penalty = -0.1
            
            final_score = base_score + importance_boost + keyword_boost + content_type_boost + length_penalty
            return min(1.0, max(0.0, final_score))
        
        # Calculate relevance scores and sort
        for result in results:
            result["relevance_score"] = calculate_relevance_score(result)
        
        # Sort by relevance score
        reranked = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
        
        logger.info(f"Reranked results. Top score: {reranked[0]['relevance_score']:.3f}" if reranked else "No results to rerank")
        return reranked[:FINAL_CONTEXT_LIMIT]
    
    def build_context(self, results: List[Dict[str, Any]]) -> str:
        """Build optimized context for Gemini"""
        if not results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results, 1):
            # Format context chunk
            metadata = result["metadata"]
            chunk_text = result["text"]
            
            # Create informative header
            header_parts = []
            if metadata.get("title"):
                header_parts.append(f"Title: {metadata['title']}")
            if metadata.get("content_category"):
                header_parts.append(f"Category: {metadata['content_category']}")
            if metadata.get("source_url"):
                header_parts.append(f"Source: {metadata['source_url']}")
            
            header = " | ".join(header_parts)
            formatted_chunk = f"[Context {i}]\n{header}\n{chunk_text}\n"
            
            # Check if adding this chunk exceeds limit
            if current_length + len(formatted_chunk) > CONTEXT_WINDOW_LIMIT:
                break
            
            context_parts.append(formatted_chunk)
            current_length += len(formatted_chunk)
        
        context = "\n".join(context_parts)
        logger.info(f"Built context with {len(context_parts)} chunks ({current_length} characters)")
        
        return context
    
    def generate_response(self, query: str, context: str, query_intent: Dict[str, Any]) -> str:
        """Generate response using Gemini 2.0 Flash"""
        try:
            # Build specialized prompt based on intent
            if "apply" in query_intent["intents"]:
                system_prompt = """You are a helpful assistant for the Directorate of Municipal Administration (DMA), Maharashtra. 
Provide detailed, step-by-step guidance for municipal services and applications. 
Focus on practical information including required documents, procedures, fees, and contact details."""
            
            elif query_intent["needs_contact"]:
                system_prompt = """You are a helpful assistant for the Directorate of Municipal Administration (DMA), Maharashtra.
Provide accurate contact information, office addresses, phone numbers, and relevant departmental details."""
            
            else:
                system_prompt = """You are a helpful assistant for the Directorate of Municipal Administration (DMA), Maharashtra.
Provide accurate, comprehensive information about municipal services, policies, and procedures based on the provided context."""
            
            # Build the complete prompt
            prompt = f"""{system_prompt}

Context Information:
{context}

User Question: {query}

Instructions:
1. Answer based ONLY on the provided context information
2. If the context doesn't contain enough information, clearly state what's missing
3. For service applications, include step-by-step procedures, required documents, and fees if available
4. For contact queries, provide complete contact details including phone numbers, addresses, and office hours if available
5. Include relevant URLs/links when provided in the context
6. If multiple options exist, present them clearly
7. Use a helpful, professional tone suitable for government services

Answer:"""

            # Generate response
            response = self.gemini_model.generate_content(prompt)
            
            if response and response.text:
                return response.text.strip()
            else:
                return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
    
    def query(self, user_query: str) -> Dict[str, Any]:
        """Process complete query pipeline"""
        start_time = time.time()
        
        # Preprocess query
        processed_query = self.preprocess_query(user_query)
        query_intent = self.extract_query_intent(processed_query)
        
        logger.info(f"Processing query: '{user_query}'")
        logger.info(f"Detected intents: {query_intent['intents']}")
        logger.info(f"Service categories: {query_intent['service_categories']}")
        
        # Build search filters
        filters = self.build_search_filters(query_intent)
        
        # Perform semantic search
        search_results = self.semantic_search(processed_query, filters)
        
        if not search_results:
            return {
                "query": user_query,
                "response": "I couldn't find relevant information for your query. Please try rephrasing or contact DMA directly for assistance.",
                "sources": [],
                "processing_time": time.time() - start_time,
                "metadata": {
                    "intent": query_intent,
                    "results_found": 0
                }
            }
        
        # Rerank results
        reranked_results = self.rerank_results(processed_query, search_results, query_intent)
        
        # Build context
        context = self.build_context(reranked_results)
        
        # Generate response
        response = self.generate_response(processed_query, context, query_intent)
        
        # Prepare sources
        sources = []
        for result in reranked_results[:5]:  # Top 5 sources
            metadata = result["metadata"]
            source = {
                "title": metadata.get("title", "Unknown"),
                "url": metadata.get("source_url"),
                "category": metadata.get("content_category", "general"),
                "relevance_score": round(result["relevance_score"], 3)
            }
            sources.append(source)
        
        return {
            "query": user_query,
            "response": response,
            "sources": sources,
            "processing_time": time.time() - start_time,
            "metadata": {
                "intent": query_intent,
                "results_found": len(search_results),
                "context_chunks": len(reranked_results)
            }
        }
    
    def initialize(self) -> bool:
        """Initialize all components"""
        logger.info("Initializing Advanced RAG Query System...")
        
        if not self.setup_embedding_model():
            return False
        
        if not self.setup_pinecone():
            return False
        
        if not self.gemini_model:
            logger.error("Gemini model not initialized")
            return False
        
        logger.info("All components initialized successfully!")
        return True

def main():
    """Interactive query interface"""
    rag = AdvancedRAGQuerySystem()
    
    if not rag.initialize():
        logger.error("Failed to initialize RAG system")
        return
    
    print("🚀 Advanced DMA RAG System Ready!")
    print("Ask questions about municipal services, applications, and procedures.")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("\n❓ Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Thank you for using DMA RAG System!")
                break
            
            if not user_input:
                continue
            
            print("\n🔍 Processing your query...")
            result = rag.query(user_input)
            
            print(f"\n📋 **Response:**")
            print(result["response"])
            
            if result["sources"]:
                print(f"\n📚 **Sources:**")
                for i, source in enumerate(result["sources"], 1):
                    print(f"{i}. {source['title']} (Relevance: {source['relevance_score']})")
                    if source['url']:
                        print(f"   🔗 {source['url']}")
            
            print(f"\n⏱️  Processing time: {result['processing_time']:.2f}s")
            print(f"📊 Results found: {result['metadata']['results_found']}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
