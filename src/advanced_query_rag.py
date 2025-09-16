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

# Configuration - Dynamic paths for deployment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "optimized_data")

# Pinecone configuration
PINECONE_INDEX_NAME = "dma-knowledge-base-multilingual-v1"

# Query optimization parameters
TOP_K_RETRIEVAL = 50  # Retrieve more candidates for reranking (increased for better article retrieval)
FINAL_CONTEXT_LIMIT = 10  # Final number of chunks to use
MIN_SIMILARITY_THRESHOLD = 0.3  # Minimum similarity for relevance (restored to original)
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
        
        # Initialize embedding model
        self.setup_embedding_model()
        
        # Initialize Pinecone
        self.setup_pinecone()
        
        # Initialize Gemini [[memory:8625932]]
        self.gemini_model = None
        self.setup_gemini()
        
    def setup_embedding_model(self) -> bool:
        """Setup E5-base-v2 embedding model for query encoding"""
        try:
            # Configure SSL to handle corporate network certificate issues
            import ssl
            import urllib3
            import os
            
            # Disable SSL warnings and verification for model download
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            # Set environment variables to disable SSL verification
            os.environ['CURL_CA_BUNDLE'] = ''
            os.environ['REQUESTS_CA_BUNDLE'] = ''
            
            # Create unverified SSL context
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Force use of multilingual-e5-base model for multilingual support
            self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-base', device="cpu")
            logger.info("Loaded multilingual-e5-base embedding model (768-dim) - supports 100+ languages including Marathi")
            
            self.model_name = "multilingual-e5-base"
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup E5-base-v2 embedding model: {e}")
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
        
        # Enhanced processing for Article queries
        article_pattern = r'\barticle\s+(\d+)\b'
        article_match = re.search(article_pattern, query_lower)
        if article_match:
            article_num = article_match.group(1)
            # For article queries, also include the specific article format used in the text
            enhanced_query = f"{query} Article {article_num}:"
            return enhanced_query
        
        return query
    
    def is_general_service_query(self, query: str) -> bool:
        """Check if query is asking about services in general"""
        query_lower = query.lower()
        
        # Keywords that indicate service application/process queries
        service_application_keywords = [
            "apply", "application", "form", "online services", 
            "how to apply", "where to apply", "registration process",
            "license application", "certificate application", "permit application"
        ]
        
        # Keywords that indicate informational queries (NOT service applications)
        informational_keywords = [
            "newly formed", "formed", "created", "established", "formation", 
            "which are", "what are", "list of", "total", "how many", "count",
            "ulb in", "urban local body", "municipal council", "municipal corporation",
            "gazette", "notification", "structure", "organization", "department"
        ]
        
        # If it's clearly an informational query, don't treat as service query
        if any(keyword in query_lower for keyword in informational_keywords):
            return False
        
        # Only treat as service query if it has clear service application keywords
        return any(keyword in query_lower for keyword in service_application_keywords)
    
    def extract_query_intent(self, query: str) -> Dict[str, Any]:
        """Extract intent and entities from query"""
        query_lower = query.lower()
        
        intent_keywords = {
            "apply": ["apply", "application", "form", "submit", "register"],
            "information": ["what", "how", "where", "when", "info", "details", "says", "do", "does"],
            "contact": ["contact", "phone", "email", "address", "office"],
            "procedure": ["procedure", "process", "steps", "how to"],
            "status": ["status", "track", "check", "progress"],
            "fee": ["fee", "cost", "charges", "payment", "amount"],
            "documents": ["documents", "papers", "requirements", "needed"],
            "article": ["article", "constitution", "law", "legal"],
            "statistics": ["how many", "total", "count", "number of", "statistics", "data", "कितने", "एकूण", "संख्या"]
        }
        
        detected_intents = []
        for intent, keywords in intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intents.append(intent)
        
        # Extract service categories with better matching logic
        service_categories = {
            "water": ["water connection", "water bill", "water supply", "plumber", "water reconnection"],
            "property": ["property tax", "property transfer", "property assessment", "property extract"],
            "trade": ["trade license", "business license", "shop license", "business registration"],
            "marriage": ["marriage registration", "marriage certificate", "wedding"],
            "birth": ["birth certificate", "birth registration"],
            "death": ["death certificate", "death registration"],
            "noc": ["meat shop noc", "meat shop", "hospital noc", "electric meter noc", "food noc", "hoarding noc", "banner noc", "tours and travels noc", "pool snooker noc", "mandap stall noc", "road digging noc", "fast food noc"],
            "approvals": ["no objection certificate", "clearance", "permission", "approval"]
        }
        
        detected_categories = []
        
        # First, check for specific multi-word phrases
        for category, keywords in service_categories.items():
            for keyword in keywords:
                if keyword in query_lower:
                    detected_categories.append(category)
                    break
        
        # If no specific matches, check for single words with context
        if not detected_categories:
            single_word_categories = {
                "water": ["water"],
                "property": ["property"],
                "trade": ["trade", "business", "shop", "license"],
                "marriage": ["marriage", "wedding"],
                "birth": ["birth"],
                "death": ["death"],
                "noc": ["noc"],
                "approvals": ["approval", "permission", "clearance"]
            }
            
            for category, keywords in single_word_categories.items():
                if any(keyword in query_lower for keyword in keywords):
                    detected_categories.append(category)
        
        return {
            "intents": detected_intents if detected_intents else ["information"],
            "service_categories": detected_categories,
            "is_procedural": any(word in query_lower for word in ["how", "procedure", "steps", "process"]),
            "needs_contact": any(word in query_lower for word in ["contact", "phone", "office", "address"]),
            "is_article_query": "article" in detected_intents or bool(re.search(r'\barticle\s+\d+\b', query_lower))
        }
    
    def handle_service_query(self, user_query: str, query_intent: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Handle service-related queries with unified link"""
        logger.info(f"Handling service query: {user_query}")
        
        # Create a unified response for all services
        response = self.generate_unified_service_response(user_query, query_intent)
        
        return {
            "query": user_query,
            "response": response,
            "sources": [{"url": "https://mahaulb.in/MahaULB/index", "title": "Maharashtra ULB Services Portal"}],
            "processing_time": time.time() - start_time,
            "metadata": {
                "intent": query_intent,
                "results_found": 1,
                "service_redirect": True
            }
        }
    
    def generate_unified_service_response(self, query: str, query_intent: Dict[str, Any]) -> str:
        """Generate a unified response for service queries"""
        query_lower = query.lower()
        
        # Detect language preference
        is_marathi = any(char in query for char in "अआइईउऊऋऌएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह")
        
        if is_marathi:
            base_response = "महाराष्ट्र शहरी स्थानिक स्वराज्य संस्थांच्या सर्व सेवांसाठी कृपया खालील लिंकवर भेट द्या:"
            portal_text = "Maharashtra ULB सेवा पोर्टल"
            help_text = "येथे तुम्हाला सर्व नगरपालिका सेवा, अर्ज प्रक्रिया, आणि आवश्यक माहिती मिळेल."
        else:
            base_response = "For all Maharashtra Urban Local Body services, please visit:"
            portal_text = "Maharashtra ULB Services Portal"
            help_text = "Here you'll find all municipal services, application processes, and required information."
        
        response = f"{base_response}\n\n🔗 **{portal_text}**: https://mahaulb.in/MahaULB/index\n\n{help_text}"
        
        # Add specific guidance based on detected service categories
        if query_intent["service_categories"]:
            categories = query_intent["service_categories"]
            if "water" in categories:
                if is_marathi:
                    response += "\n\n💧 पाणी कनेक्शन, बिल पेमेंट आणि संबंधित सेवांसाठी या पोर्टलवरील 'Water Services' विभाग पहा."
                else:
                    response += "\n\n💧 For water connection, bill payment and related services, check the 'Water Services' section on the portal."
            
            elif "property" in categories:
                if is_marathi:
                    response += "\n\n🏠 मालमत्ता कर, हस्तांतरण आणि संबंधित सेवांसाठी या पोर्टलवरील 'Property Services' विभाग पहा."
                else:
                    response += "\n\n🏠 For property tax, transfer and related services, check the 'Property Services' section on the portal."
            
            elif "trade" in categories:
                if is_marathi:
                    response += "\n\n🏪 व्यापार परवाना, नवीकरण आणि संबंधित सेवांसाठी या पोर्टलवरील 'Trade License' विभाग पहा."
                else:
                    response += "\n\n🏪 For trade license, renewal and related services, check the 'Trade License' section on the portal."
        
        return response
    
    def build_search_filters(self, query_intent: Dict[str, Any]) -> Dict[str, Any]:
        """Build metadata filters for targeted search"""
        filters = {}
        
        # For article queries, don't apply restrictive filters
        if query_intent.get("is_article_query", False):
            # Only apply basic language filter if needed, but keep search broad
            return {}
        
        # Filter by service categories if detected
        if query_intent["service_categories"]:
            category_conditions = []
            
            # Map our categories to actual content categories in the data
            category_mapping = {
                "water": ["water_services"],
                "property": ["property_services"], 
                "trade": ["trade_license"],
                "marriage": ["vital_records"],
                "birth": ["vital_records"],
                "death": ["vital_records"],
                "noc": ["trade_license", "approvals_noc"],  # NOCs can be in both categories
                "approvals": ["approvals_noc"]
            }
            
            for category in query_intent["service_categories"]:
                mapped_categories = category_mapping.get(category, [category])
                for mapped_category in mapped_categories:
                    category_conditions.append({"content_category": {"$eq": mapped_category}})
            
            if len(category_conditions) == 1:
                filters.update(category_conditions[0])
            else:
                filters["$or"] = category_conditions
        
        # Boost service-related content for application queries, but don't make it mandatory
        # Removed the mandatory service_related filter as it was excluding articles
        
        # Filter for contact information
        if query_intent["needs_contact"]:
            filters["$or"] = [
                {"content_category": {"$eq": "organizational"}},
                {"content_category": {"$eq": "office_contact"}},
                {"content_category": {"$eq": "contact_details"}},
                {"chunk_type": {"$eq": "table_row"}},
                {"chunk_type": {"$eq": "contact_info"}},
                {"keywords": {"$in": ["contact", "phone", "address"]}}
            ]
        
        return filters
    
    def semantic_search(self, query: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform semantic search with filtering"""
        try:
            # Check if embedding model is available
            if self.embedding_model is None:
                logger.error("Embedding model not initialized")
                return []
            
            # Encode the query with query prefix (required for multilingual-e5-base)
            query_with_prefix = f"query: {query}"
            query_embedding = self.embedding_model.encode(query_with_prefix, normalize_embeddings=True).tolist()
            
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
            
            # Special boost for article number matches
            article_boost = 0.0
            article_pattern = r'\barticle\s+(\d+)\b'
            query_article_match = re.search(article_pattern, query_lower)
            if query_article_match:
                article_num = query_article_match.group(1)
                if f"article {article_num}" in text.lower():
                    article_boost = 0.5  # Strong boost for exact article matches
            
            # Boost based on content type preferences
            content_type_boost = 0.0
            content_category = metadata.get("content_category", "")
            
            if "apply" in query_intent["intents"] and "service" in content_category:
                content_type_boost += 0.3
            
            if query_intent["needs_contact"] and content_category in ["organizational", "office_contact", "contact_details"]:
                content_type_boost += 0.4
            
            if query_intent["is_procedural"] and "procedure" in text:
                content_type_boost += 0.2
            
            # Boost for statistical queries asking for totals
            if "statistics" in query_intent["intents"] and metadata.get("chunk_type") == "total_statistics":
                content_type_boost += 0.6  # Strong boost for total statistics
            
            if "statistics" in query_intent["intents"] and "total" in metadata.get("title", "").lower():
                content_type_boost += 0.4
            
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
            
            final_score = base_score + importance_boost + keyword_boost + content_type_boost + length_penalty + article_boost
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
            # Check for greeting patterns first
            greeting_patterns = ["hi", "hello", "hey", "namaste", "namaskar", "good morning", "good afternoon", "good evening"]
            if any(pattern in query.lower() for pattern in greeting_patterns) and len(query.split()) <= 3:
                return "🙏 Namaste! How may I assist you with municipal services today?"
            
            # Check if query is about services without exact match
            if not context.strip() or len(context.strip()) < 50:
                return self.handle_no_context_response(query)
            
            # Build specialized prompt based on intent
            if "apply" in query_intent["intents"]:
                system_prompt = """You are a helpful assistant for DMA, Maharashtra. Provide concise, helpful guidance for municipal services. Be polite and direct."""
            
            elif query_intent["needs_contact"]:
                system_prompt = """You are a helpful assistant for DMA, Maharashtra. Provide accurate contact information concisely and politely."""
            
            else:
                system_prompt = """You are a helpful assistant for DMA, Maharashtra. Provide accurate, concise information about municipal services. Be polite and helpful."""
            
            # Build the complete prompt
            prompt = f"""{system_prompt}

Context Information:
{context}

User Question: {query}

Instructions:
1. Be humble, respectful, and concise (maximum 3-4 sentences unless absolutely necessary)
2. Answer based ONLY on the provided context information
3. Use humble language: "I can help you with...", "Let me assist you...", "Here's what I found..."
4. If information is incomplete, politely mention what's missing: "🙏 Sorry, I don't have complete information about..."
5. For service applications, provide clear steps without excessive detail
6. Include relevant URLs/links when available
7. Keep responses direct and user-focused
8. Avoid listing multiple services unless specifically asked

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
    
    def handle_no_context_response(self, query: str) -> str:
        """Handle queries when no relevant context is found"""
        # Try to suggest related services with positive framing
        return self.suggest_helpful_alternatives(query)
    
    def suggest_helpful_alternatives(self, query: str) -> str:
        """Suggest helpful alternatives when specific information is not available"""
        try:
            import json
            services_file = os.path.join(BASE_DIR, "final_data", "services.json")
            
            if not os.path.exists(services_file):
                return "I'd be happy to help you further! For specific information about your query, please contact DMA directly at comdir.ud@maharashtra.gov.in or visit https://mahadma.maharashtra.gov.in/ for comprehensive information."
            
            with open(services_file, 'r', encoding='utf-8') as f:
                services_data = json.load(f)
            
            # Extract keywords from query
            query_lower = query.lower()
            query_keywords = []
            
            # Service-related keywords mapping
            service_keywords = {
                "noc": ["noc", "clearance", "approval", "permission"],
                "license": ["license", "permit", "registration", "renewal"],
                "water": ["water", "connection", "bill", "payment", "plumber"],
                "property": ["property", "tax", "ownership", "transfer"],
                "marriage": ["marriage", "wedding", "registration", "certificate"],
                "trade": ["trade", "business", "shop", "commercial"],
                "hospital": ["hospital", "medical", "healthcare", "clinic"],
                "electric": ["electric", "electricity", "meter", "power"],
                "food": ["food", "restaurant", "hotel", "catering"]
            }
            
            # Find matching categories
            matching_categories = []
            for category, keywords in service_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    matching_categories.append(category)
            
            # Find related services
            related_services = []
            for service in services_data:
                service_name = service.get('service', '').lower()
                service_link = service.get('link', '')
                
                # Check if service matches any category or contains query keywords
                if matching_categories:
                    for category in matching_categories:
                        if category in service_name:
                            related_services.append({
                                'name': service.get('service', ''),
                                'link': service_link
                            })
                            break
                else:
                    # Fallback: check for direct keyword matches
                    query_words = query_lower.split()
                    if any(word in service_name for word in query_words if len(word) > 2):
                        related_services.append({
                            'name': service.get('service', ''),
                            'link': service_link
                        })
            
            # Remove duplicates and limit results
            unique_services = []
            seen_names = set()
            for service in related_services:
                if service['name'] not in seen_names:
                    unique_services.append(service)
                    seen_names.add(service['name'])
                if len(unique_services) >= 5:  # Limit to 5 suggestions
                    break
            
            if unique_services:
                response = "While I don't have specific details about your query, I found these related DMA services that might be helpful:\\n\\n"
                for i, service in enumerate(unique_services, 1):
                    response += f"{i}. **{service['name']}**"
                    if service['link']:
                        response += f" - {service['link']}"
                    response += "\\n"
                response += "\\nIf none of these match what you're looking for, please contact DMA directly at comdir.ud@maharashtra.gov.in for personalized assistance!"
                return response
            else:
                return "I'd be happy to help you with your query! For specific information, please contact DMA directly at comdir.ud@maharashtra.gov.in or visit https://mahadma.maharashtra.gov.in/ where you can find comprehensive information about all municipal services."
                
        except Exception as e:
            logger.error(f"Error suggesting related services: {e}")
            return "I'd be happy to help you further! Please contact DMA directly at comdir.ud@maharashtra.gov.in or visit https://mahadma.maharashtra.gov.in/ for comprehensive assistance with your query."
    
    def preprocess_query_with_context(self, user_query: str, conversation_context: str = "") -> str:
        """Enhanced query preprocessing that considers conversation context"""
        processed_query = self.preprocess_query(user_query)
        
        if not conversation_context:
            return processed_query
        
        # Check for reference words that might need context
        reference_words = ['that', 'it', 'this', 'they', 'them', 'those', 'these', 'previous', 'above', 'earlier']
        query_lower = user_query.lower()
        
        has_reference = any(word in query_lower for word in reference_words)
        
        # If query is very short or has reference words, enhance with context
        if len(user_query.split()) <= 3 or has_reference:
            # Extract relevant context from conversation
            context_lines = conversation_context.split('\n')
            recent_context = context_lines[-4:] if len(context_lines) > 4 else context_lines
            
            # Add context information to help understanding
            enhanced_query = f"{processed_query}\n\nConversation context:\n{chr(10).join(recent_context)}"
            return enhanced_query
        
        return processed_query
    
    def generate_response_with_context(self, query: str, context: str, query_intent: Dict[str, Any], conversation_context: str = "") -> str:
        """Generate response using Gemini 2.0 Flash with conversation awareness"""
        try:
            # Check for greeting patterns first
            greeting_patterns = ["hi", "hello", "hey", "namaste", "namaskar", "good morning", "good afternoon", "good evening"]
            if any(pattern in query.lower() for pattern in greeting_patterns) and len(query.split()) <= 3:
                return "🙏 Namaste! How may I assist you with municipal services today?"
            
            # Check if query is about services without exact match
            if not context.strip() or len(context.strip()) < 50:
                return self.handle_no_context_response(query)
            
            # Build specialized prompt based on intent and query type
            query_lower = query.lower()
            
            # Special handling for news/announcements queries
            if any(keyword in query_lower for keyword in ["news", "announcement", "update", "latest"]):
                system_prompt = """You are a helpful assistant for DMA, Maharashtra. When users ask for news, announcements, or updates, provide BOTH News and Announcements links:

For English users, provide:
- News: https://mahadma.maharashtra.gov.in/en/news-2/ 
- Announcements: https://mahadma.maharashtra.gov.in/en/category/announcements/

For Marathi users, also include:
- Marathi News: https://mahadma.maharashtra.gov.in/news/ 
- Marathi Announcements: https://mahadma.maharashtra.gov.in/category/announcements-mr/

Always mention both pages as they contain different types of information. Be helpful and comprehensive."""
            
            elif "apply" in query_intent["intents"]:
                system_prompt = """You are a helpful assistant for DMA, Maharashtra. Provide concise, helpful guidance for municipal services. Be polite and direct."""
            
            elif query_intent["needs_contact"]:
                system_prompt = """You are a helpful assistant for DMA, Maharashtra. Provide accurate contact information concisely and politely."""
            
            else:
                system_prompt = """You are a helpful assistant for DMA, Maharashtra. Provide accurate, concise information about municipal services. Be polite and helpful."""
            
            # Build the complete prompt with conversation awareness
            prompt = f"""{system_prompt}

Context Information:
{context}"""

            # Add conversation context if available
            if conversation_context:
                prompt += f"""

Previous Conversation:
{conversation_context}"""

            prompt += f"""

User Question: {query}

Instructions:
1. Be polite and concise (maximum 3-4 sentences unless absolutely necessary)
2. Answer based ONLY on the provided context information
3. Use natural language: "There are...", "The address is...", "Here's the information..."
4. If this is a follow-up question, refer to the previous conversation appropriately
5. For reference words like "that", "it", "this", use the conversation context to understand what they refer to
6. If information is incomplete, offer helpful alternatives: "I have some information about this, but for complete details, please contact DMA at comdir.ud@maharashtra.gov.in"
7. For service applications, provide clear steps without excessive detail
8. Include relevant URLs/links when available
9. Keep responses direct and user-focused
10. Avoid listing multiple services unless specifically asked
11. CRITICAL: For numerical/statistical queries (total, count, how many), provide the DIRECT ANSWER FIRST, then brief context if needed
12. CRITICAL: When user asks for totals/counts, prioritize the final number over detailed breakdowns
13. CRITICAL: Respond with PLAIN TEXT ONLY. DO NOT use JSON format, markdown code blocks, or any special formatting.
14. CRITICAL: Do not wrap your response in ```json``` or any code blocks.

Provide a direct, plain text answer:"""

            # Generate response
            response = self.gemini_model.generate_content(prompt)
            
            if response and response.text:
                # Clean up the response to remove any JSON formatting
                clean_response = self.clean_response_format(response.text.strip())
                return clean_response
            else:
                return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
    
    def clean_response_format(self, response_text: str) -> str:
        """Clean response text to remove JSON formatting and code blocks"""
        try:
            # Remove markdown code blocks (```json, ```, etc.)
            response_text = re.sub(r'```[\w]*\n?', '', response_text)
            response_text = re.sub(r'```', '', response_text)
            
            # Try to extract content from JSON structure if present
            if response_text.strip().startswith('{') and response_text.strip().endswith('}'):
                try:
                    import json
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
            logger.warning(f"Error cleaning response format: {e}")
            return response_text
    
    def query(self, user_query: str, conversation_context: str = "") -> Dict[str, Any]:
        """Process complete query pipeline"""
        start_time = time.time()
        
        # Preprocess query with conversation context
        processed_query = self.preprocess_query_with_context(user_query, conversation_context)
        query_intent = self.extract_query_intent(processed_query)
        
        logger.info(f"Processing query: '{user_query}'")
        logger.info(f"Detected intents: {query_intent['intents']}")
        logger.info(f"Service categories: {query_intent['service_categories']}")
        
        # Check if this is a service-related query (but not informational ULB queries)
        is_service_query = self.is_general_service_query(user_query)
        has_service_categories = bool(query_intent["service_categories"])
        
        # Additional check: don't redirect informational ULB queries to services
        query_lower = user_query.lower()
        is_ulb_info_query = any(keyword in query_lower for keyword in [
            "newly formed ulb", "formed ulb", "which ulb", "list of ulb", 
            "ulb in maharashtra", "urban local body", "municipal council",
            "how many ulb", "total ulb", "count of ulb"
        ])
        
        if (is_service_query or has_service_categories) and not is_ulb_info_query:
            return self.handle_service_query(user_query, query_intent, start_time)
        
        # Build search filters
        filters = self.build_search_filters(query_intent)
        
        # Perform semantic search
        search_results = self.semantic_search(processed_query, filters)
        
        if not search_results:
            # Try to suggest helpful alternatives with positive framing
            suggested_response = self.suggest_helpful_alternatives(processed_query)
            return {
                "query": user_query,
                "response": suggested_response,
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
        
        # Generate response with conversation context
        response = self.generate_response_with_context(processed_query, context, query_intent, conversation_context)
        
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
