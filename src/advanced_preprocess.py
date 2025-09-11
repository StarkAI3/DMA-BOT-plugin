#!/usr/bin/env python3
"""
Advanced Data Preprocessing for High-Accuracy RAG System
Optimized for semantic coherence and retrieval accuracy
"""

import os
import json
import glob
import hashlib
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Iterable, Tuple, Optional
from dataclasses import dataclass, asdict
import nltk
from sentence_transformers import SentenceTransformer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Configuration
BASE_DIR = "/home/stark/Desktop/DMA_BOT"
FINAL_DATA_DIR = os.path.join(BASE_DIR, "final_data")
NORMAL_DATA_DIR = os.path.join(FINAL_DATA_DIR, "normal data")
TABULAR_DATA_DIR = os.path.join(FINAL_DATA_DIR, "tabular data")
SERVICES_FILE = os.path.join(FINAL_DATA_DIR, "services.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "optimized_data")

# Embedding model for semantic chunking [[memory:5458554]]
EMBEDDING_MODEL = "models/models--dunzhang--stella_en_400M_v5"

# Advanced chunking parameters
MAX_CHUNK_SIZE = 512  # Optimal for embedding models
MIN_CHUNK_SIZE = 100
SEMANTIC_THRESHOLD = 0.5  # Cosine similarity threshold for chunk coherence
CHUNK_OVERLAP = 50

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Enhanced metadata for better retrieval context"""
    data_type: str
    content_category: str
    source_url: Optional[str]
    title: str
    language: str
    chunk_type: str
    semantic_keywords: List[str]
    file_source: str
    section_hierarchy: List[str]
    entity_types: List[str]
    importance_score: float
    has_structured_data: bool
    date_mentioned: Optional[str]
    location_mentioned: Optional[str]
    department_mentioned: Optional[str]
    service_related: bool

class AdvancedPreprocessor:
    """Advanced preprocessing with semantic-aware chunking"""
    
    def __init__(self):
        self.embedding_model = None
        self.ensure_dirs()
        self.stats = {
            'total_chunks': 0,
            'web_content_chunks': 0,
            'service_chunks': 0,
            'tabular_chunks': 0,
            'high_quality_chunks': 0
        }
    
    def ensure_dirs(self) -> None:
        """Create necessary directories"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def load_embedding_model(self) -> None:
        """Load E5-base-v2 embedding model for semantic chunking"""
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
            
            self.embedding_model = SentenceTransformer('intfloat/e5-base-v2', device='cpu')
            logger.info("Loaded E5-base-v2 embedding model (768-dim)")
        except Exception as e:
            logger.error(f"Failed to load E5-base-v2 embedding model: {e}")
            self.embedding_model = None
            raise
    
    def hash_id(self, *parts: str) -> str:
        """Generate stable hash ID"""
        m = hashlib.md5()
        for p in parts:
            m.update(str(p).encode("utf-8", errors="ignore"))
        return m.hexdigest()
    
    def clean_text(self, text: str) -> str:
        """Advanced text cleaning"""
        if not text:
            return ""
        
        # Remove HTML artifacts and normalize whitespace
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Fix common OCR/scraping errors
        text = re.sub(r'(\w)([A-Z])', r'\1 \2', text)  # Add space before capitals
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # camelCase fixing
        
        # Clean up multiple spaces and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract key entities for better metadata"""
        entities = {
            'departments': [],
            'locations': [],
            'dates': [],
            'services': [],
            'officials': []
        }
        
        # Department patterns
        dept_patterns = [
            r'directorate of municipal administration',
            r'municipal corporation',
            r'municipal council',
            r'nagar panchayat',
            r'urban development',
            r'DMA'
        ]
        
        # Location patterns (Maharashtra districts/cities)
        location_patterns = [
            r'maharashtra', r'mumbai', r'pune', r'nagpur', r'nashik',
            r'aurangabad', r'solapur', r'kolhapur', r'sangli', r'satara'
        ]
        
        # Service patterns
        service_patterns = [
            r'water connection', r'trade license', r'property tax',
            r'marriage registration', r'noc', r'birth certificate',
            r'death certificate', r'building plan'
        ]
        
        text_lower = text.lower()
        
        for pattern in dept_patterns:
            if re.search(pattern, text_lower):
                entities['departments'].append(pattern.replace(r'\b', '').replace(r'\\b', ''))
        
        for pattern in location_patterns:
            if re.search(pattern, text_lower):
                entities['locations'].append(pattern)
        
        for pattern in service_patterns:
            if re.search(pattern, text_lower):
                entities['services'].append(pattern.replace(r'\b', '').replace(r'\\b', ''))
        
        # Extract dates
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text_lower)
            entities['dates'].extend(matches)
        
        return entities
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract semantic keywords for better retrieval"""
        # Common important terms in municipal administration
        important_terms = [
            'application', 'certificate', 'license', 'registration', 'approval',
            'tax', 'fee', 'payment', 'renewal', 'transfer', 'correction',
            'municipal', 'council', 'corporation', 'panchayat', 'urban',
            'water', 'property', 'trade', 'business', 'marriage', 'birth',
            'death', 'noc', 'building', 'plan', 'permission'
        ]
        
        text_lower = text.lower()
        found_keywords = []
        
        for term in important_terms:
            if term in text_lower:
                found_keywords.append(term)
        
        return found_keywords[:10]  # Limit to top 10 keywords
    
    def calculate_importance_score(self, text: str, metadata: Dict[str, Any]) -> float:
        """Calculate importance score for ranking"""
        score = 0.5  # Base score
        
        # Higher score for service-related content
        if metadata.get('service_related', False):
            score += 0.3
        
        # Higher score for structured content
        if metadata.get('has_structured_data', False):
            score += 0.2
        
        # Score based on content length (neither too short nor too long)
        text_len = len(text)
        if 200 <= text_len <= 800:
            score += 0.2
        elif text_len < 100:
            score -= 0.3
        
        # Higher score for content with contact information
        if any(term in text.lower() for term in ['contact', 'phone', 'email', 'address']):
            score += 0.1
        
        # Higher score for procedural content
        if any(term in text.lower() for term in ['step', 'procedure', 'process', 'how to', 'apply']):
            score += 0.2
        
        return min(1.0, max(0.0, score))
    
    def semantic_split(self, text: str, max_size: int = MAX_CHUNK_SIZE) -> List[str]:
        """Split text using semantic boundaries"""
        if len(text) <= max_size:
            return [text]
        
        # Try sentence-based splitting first
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            # Fallback to simple period splitting
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed limit
            if len(current_chunk) + len(sentence) + 1 <= max_size:
                current_chunk = (current_chunk + " " + sentence).strip()
            else:
                # Save current chunk if it's substantial
                if len(current_chunk) >= MIN_CHUNK_SIZE:
                    chunks.append(current_chunk)
                    # Start new chunk with overlap
                    if CHUNK_OVERLAP > 0 and len(current_chunk) > CHUNK_OVERLAP:
                        overlap_text = current_chunk[-CHUNK_OVERLAP:].strip()
                        current_chunk = overlap_text + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk += " " + sentence
                
                # If single sentence is too long, force split
                if len(current_chunk) > max_size:
                    if chunks:  # Only if we have previous chunks
                        chunks.append(current_chunk[:max_size])
                        current_chunk = current_chunk[max_size-CHUNK_OVERLAP:]
                    else:
                        current_chunk = sentence  # Keep the long sentence as is
        
        # Add remaining chunk
        if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE:
            chunks.append(current_chunk)
        
        return chunks if chunks else [text]  # Fallback to original text
    
    def process_web_content(self) -> List[Dict[str, Any]]:
        """Process web content files with advanced chunking"""
        logger.info("Processing web content files...")
        records = []
        
        for file_path in sorted(glob.glob(os.path.join(NORMAL_DATA_DIR, "*.json"))):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not isinstance(data, dict):
                    continue
                
                file_info = data.get("file_info", {})
                raw_data = data.get("raw_data", {})
                
                url = raw_data.get("url", "")
                title = raw_data.get("title", file_info.get("filename", ""))
                content = self.clean_text(raw_data.get("content", ""))
                links = raw_data.get("links", [])
                
                if not content:
                    continue
                
                # Extract section hierarchy
                sections = self.extract_sections(content)
                
                # Semantic chunking
                chunks = self.semantic_split(content)
                
                for idx, chunk in enumerate(chunks):
                    if len(chunk) < MIN_CHUNK_SIZE:
                        continue
                    
                    entities = self.extract_entities(chunk)
                    keywords = self.extract_keywords(chunk)
                    
                    # Determine content category
                    category = self.categorize_content(chunk, title, url)
                    
                    metadata = ChunkMetadata(
                        data_type="web_content",
                        content_category=category,
                        source_url=url,
                        title=title,
                        language=self.detect_language(chunk),
                        chunk_type="semantic_content",
                        semantic_keywords=keywords,
                        file_source=os.path.basename(file_path),
                        section_hierarchy=sections,
                        entity_types=list(entities.keys()),
                        importance_score=0.0,  # Will be calculated
                        has_structured_data=bool(links),
                        date_mentioned=entities['dates'][0] if entities['dates'] else None,
                        location_mentioned=entities['locations'][0] if entities['locations'] else None,
                        department_mentioned=entities['departments'][0] if entities['departments'] else None,
                        service_related=bool(entities['services'])
                    )
                    
                    metadata.importance_score = self.calculate_importance_score(chunk, asdict(metadata))
                    
                    record = {
                        "id": self.hash_id(file_path, "content", str(idx)),
                        "text": chunk,
                        "metadata": asdict(metadata)
                    }
                    
                    records.append(record)
                    self.stats['web_content_chunks'] += 1
                    
                    if metadata.importance_score > 0.7:
                        self.stats['high_quality_chunks'] += 1
                
                # Process links separately for comprehensive coverage
                if links:
                    self.process_links(links, file_path, title, url, records)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        logger.info(f"Processed {len(records)} web content chunks")
        return records
    
    def process_links(self, links: List[Dict], file_path: str, title: str, url: str, records: List[Dict]):
        """Process links as separate knowledge items"""
        link_texts = []
        for link in links:
            text = link.get("text", "")
            link_url = link.get("url", "")
            if text and link_url:
                link_texts.append(f"{text}: {link_url}")
        
        if link_texts:
            links_content = "\n".join(link_texts)
            
            metadata = ChunkMetadata(
                data_type="web_content",
                content_category="navigation_links",
                source_url=url,
                title=title,
                language="en",
                chunk_type="links_collection",
                semantic_keywords=self.extract_keywords(links_content),
                file_source=os.path.basename(file_path),
                section_hierarchy=["links"],
                entity_types=["services"],
                importance_score=0.6,
                has_structured_data=True,
                date_mentioned=None,
                location_mentioned=None,
                department_mentioned=None,
                service_related=True
            )
            
            record = {
                "id": self.hash_id(file_path, "links"),
                "text": links_content,
                "metadata": asdict(metadata)
            }
            
            records.append(record)
    
    def extract_sections(self, content: str) -> List[str]:
        """Extract document sections for hierarchy"""
        sections = []
        
        # Look for section headers (capitalized lines, numbered sections, etc.)
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.isupper() or 
                        re.match(r'^\d+\.', line) or
                        re.match(r'^[A-Z][^.]*$', line) and len(line) < 100):
                sections.append(line)
        
        return sections[:5]  # Limit to top 5 sections
    
    def categorize_content(self, text: str, title: str, url: str) -> str:
        """Categorize content for better organization"""
        text_lower = text.lower()
        title_lower = title.lower()
        url_lower = url.lower()
        
        # Service-related content
        if any(term in text_lower for term in ['application', 'form', 'apply', 'service', 'license']):
            return "services"
        
        # News and announcements
        if any(term in title_lower for term in ['news', 'announcement', 'notice']):
            return "news_announcements"
        
        # Policies and regulations
        if any(term in text_lower for term in ['policy', 'rule', 'regulation', 'act', 'amendment']):
            return "policies_regulations"
        
        # Contact and organizational
        if any(term in text_lower for term in ['contact', 'office', 'department', 'structure']):
            return "organizational"
        
        # Schemes and programs
        if any(term in text_lower for term in ['scheme', 'program', 'mission', 'yojana']):
            return "schemes_programs"
        
        return "general_information"
    
    def detect_language(self, text: str) -> str:
        """Detect language (English or Marathi)"""
        # Simple Devanagari detection
        for char in text:
            if '\u0900' <= char <= '\u097F':
                return "mr"
        return "en"
    
    def process_tabular_data(self) -> List[Dict[str, Any]]:
        """Process tabular data with enhanced structure preservation"""
        logger.info("Processing tabular data...")
        records = []
        
        for file_path in sorted(glob.glob(os.path.join(TABULAR_DATA_DIR, "*.json"))):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    # Process each row
                    for idx, row in enumerate(data):
                        if isinstance(row, dict):
                            row_text = self.format_table_row(row)
                            
                            if len(row_text) < MIN_CHUNK_SIZE:
                                continue
                            
                            entities = self.extract_entities(row_text)
                            keywords = self.extract_keywords(row_text)
                            
                            metadata = ChunkMetadata(
                                data_type="tabular_data",
                                content_category="structured_data",
                                source_url=None,
                                title=os.path.basename(file_path).replace('.json', ''),
                                language="en",
                                chunk_type="table_row",
                                semantic_keywords=keywords,
                                file_source=os.path.basename(file_path),
                                section_hierarchy=["tabular_data"],
                                entity_types=list(entities.keys()),
                                importance_score=0.7,  # Tables are generally important
                                has_structured_data=True,
                                date_mentioned=entities['dates'][0] if entities['dates'] else None,
                                location_mentioned=entities['locations'][0] if entities['locations'] else None,
                                department_mentioned=entities['departments'][0] if entities['departments'] else None,
                                service_related=bool(entities['services'])
                            )
                            
                            record = {
                                "id": self.hash_id(file_path, "row", str(idx)),
                                "text": row_text,
                                "metadata": asdict(metadata)
                            }
                            
                            records.append(record)
                            self.stats['tabular_chunks'] += 1
                            self.stats['high_quality_chunks'] += 1
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        logger.info(f"Processed {len(records)} tabular data chunks")
        return records
    
    def format_table_row(self, row: Dict[str, Any]) -> str:
        """Format table row for optimal retrieval"""
        parts = []
        
        # Handle common column names with better formatting
        column_priority = [
            'name', 'title', 'service', 'department', 'district', 'division',
            'contact', 'phone', 'email', 'address', 'description'
        ]
        
        # Add priority columns first
        for col in column_priority:
            for key, value in row.items():
                if col.lower() in key.lower() and str(value).strip():
                    parts.append(f"{key}: {str(value).strip()}")
                    break
        
        # Add remaining columns
        for key, value in row.items():
            formatted_key = key.replace('_', ' ').replace('-', ' ').title()
            value_str = str(value).strip()
            
            if value_str and f"{key}:" not in ' '.join(parts):
                parts.append(f"{formatted_key}: {value_str}")
        
        return " | ".join(parts)
    
    def process_services(self) -> List[Dict[str, Any]]:
        """Process services with enhanced metadata"""
        logger.info("Processing services data...")
        records = []
        
        try:
            with open(SERVICES_FILE, 'r', encoding='utf-8') as f:
                services_data = json.load(f)
            
            for idx, service in enumerate(services_data):
                service_name = service.get('service', '')
                service_link = service.get('link', '')
                
                if not service_name:
                    continue
                
                # Create comprehensive service text
                service_text = f"Service: {service_name}"
                if service_link:
                    service_text += f" | Link: {service_link}"
                
                # Add service category
                category = self.categorize_service(service_name)
                service_text += f" | Category: {category}"
                
                entities = self.extract_entities(service_text)
                keywords = self.extract_keywords(service_text)
                
                metadata = ChunkMetadata(
                    data_type="service",
                    content_category=category,
                    source_url=service_link,
                    title=service_name,
                    language="en",
                    chunk_type="service_definition",
                    semantic_keywords=keywords + [service_name.lower()],
                    file_source="services.json",
                    section_hierarchy=["services", category],
                    entity_types=["services"],
                    importance_score=0.9,  # Services are very important
                    has_structured_data=True,
                    date_mentioned=None,
                    location_mentioned=None,
                    department_mentioned="DMA",
                    service_related=True
                )
                
                record = {
                    "id": self.hash_id("service", service_name, service_link),
                    "text": service_text,
                    "metadata": asdict(metadata)
                }
                
                records.append(record)
                self.stats['service_chunks'] += 1
                self.stats['high_quality_chunks'] += 1
        
        except Exception as e:
            logger.error(f"Error processing services: {e}")
        
        logger.info(f"Processed {len(records)} service chunks")
        return records
    
    def categorize_service(self, service_name: str) -> str:
        """Categorize services for better organization"""
        name_lower = service_name.lower()
        
        if any(term in name_lower for term in ['marriage', 'birth', 'death']):
            return "vital_records"
        elif any(term in name_lower for term in ['water', 'connection', 'bill']):
            return "water_services"
        elif any(term in name_lower for term in ['trade', 'license', 'business']):
            return "trade_license"
        elif any(term in name_lower for term in ['property', 'tax']):
            return "property_services"
        elif any(term in name_lower for term in ['noc', 'permission', 'approval']):
            return "approvals_noc"
        else:
            return "general_services"
    
    def build_manifest(self) -> Dict[str, Any]:
        """Build comprehensive processing manifest"""
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "preprocessing_version": "2.0_advanced",
            "input_sources": {
                "normal_data": NORMAL_DATA_DIR,
                "tabular_data": TABULAR_DATA_DIR,
                "services_data": SERVICES_FILE
            },
            "processing_stats": self.stats,
            "chunking_strategy": {
                "method": "semantic_aware",
                "max_chunk_size": MAX_CHUNK_SIZE,
                "min_chunk_size": MIN_CHUNK_SIZE,
                "overlap_size": CHUNK_OVERLAP,
                "uses_embeddings": self.embedding_model is not None
            },
            "quality_metrics": {
                "total_chunks": self.stats['total_chunks'],
                "high_quality_percentage": round(
                    (self.stats['high_quality_chunks'] / max(1, self.stats['total_chunks'])) * 100, 2
                )
            }
        }
    
    def run(self) -> None:
        """Execute the complete preprocessing pipeline"""
        logger.info("Starting advanced preprocessing pipeline...")
        
        # Load embedding model for semantic processing
        self.load_embedding_model()
        
        # Process all data sources
        web_records = self.process_web_content()
        tabular_records = self.process_tabular_data()
        service_records = self.process_services()
        
        # Combine all records
        all_records = web_records + tabular_records + service_records
        self.stats['total_chunks'] = len(all_records)
        
        # Write output files
        output_files = {
            'web_content_chunks.jsonl': web_records,
            'tabular_data_chunks.jsonl': tabular_records,
            'service_chunks.jsonl': service_records,
            'all_chunks.jsonl': all_records
        }
        
        for filename, records in output_files.items():
            output_path = os.path.join(OUTPUT_DIR, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            logger.info(f"Written {len(records)} records to {filename}")
        
        # Write manifest
        manifest = self.build_manifest()
        manifest_path = os.path.join(OUTPUT_DIR, 'manifest.json')
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        logger.info("Advanced preprocessing completed successfully!")
        logger.info(f"Total chunks: {self.stats['total_chunks']}")
        logger.info(f"High quality chunks: {self.stats['high_quality_chunks']} ({manifest['quality_metrics']['high_quality_percentage']}%)")
        
        print(json.dumps(manifest, ensure_ascii=False, indent=2))

def main():
    """Main execution function"""
    preprocessor = AdvancedPreprocessor()
    preprocessor.run()

if __name__ == "__main__":
    main()
