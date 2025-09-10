#!/usr/bin/env python3
"""
Advanced Embedding and Vector Database Upsert System
Optimized for hardware constraints and maximum retrieval accuracy
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Generator, Optional, Tuple
import numpy as np
from tqdm import tqdm
import gc
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Vector database and embedding imports
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone
import torch

# Configuration
BASE_DIR = "/home/stark/Desktop/DMA_BOT"
DATA_DIR = os.path.join(BASE_DIR, "optimized_data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Hardware optimized settings [[memory:8625932]]
BATCH_SIZE = 8  # Reduced for 12GB RAM constraint
MAX_MEMORY_USAGE = 0.7  # Use 70% of available RAM
EMBEDDING_DIM = 1024  # Stella model dimension
DEVICE = "cpu"  # Intel i3 CPU optimization

# Pinecone configuration
PINECONE_INDEX_NAME = "dma-knowledge-base-v3"  # Changed to v3 to avoid old data
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedEmbeddingPipeline:
    """Advanced embedding pipeline optimized for accuracy and hardware constraints"""
    
    def __init__(self):
        self.model = None
        self.pc = None
        self.index = None
        self.model_name = None
        self.stats = {
            'total_vectors': 0,
            'successful_upserts': 0,
            'failed_upserts': 0,
            'processing_time': 0,
            'avg_embedding_time': 0
        }
    
    def setup_model(self) -> bool:
        """Setup E5-base-v2 embedding model with hardware optimization"""
        try:
            self.model = SentenceTransformer('intfloat/e5-base-v2', device=DEVICE)
            self.model_name = "e5-base-v2"
            logger.info("Loaded E5-base-v2 embedding model (768-dim)")
            
            # Optimize model for CPU inference
            if DEVICE == "cpu":
                self.model.eval()
                # Enable CPU optimizations
                torch.set_num_threads(4)  # Optimize for Intel i3
            
            # Test model with a sample text
            test_embedding = self.model.encode("test text", show_progress_bar=False)
            global EMBEDDING_DIM
            EMBEDDING_DIM = len(test_embedding)
            
            logger.info(f"Model loaded successfully. Embedding dimension: {EMBEDDING_DIM}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup embedding model: {e}")
            return False
    
    def setup_pinecone(self) -> bool:
        """Setup Pinecone connection and index"""
        try:
            # Initialize Pinecone
            api_key = os.getenv('PINECONE_API_KEY')
            if not api_key:
                logger.error("PINECONE_API_KEY environment variable not set")
                return False
            
            self.pc = Pinecone(api_key=api_key)
            
            # Check if index exists, create if not
            if PINECONE_INDEX_NAME not in [index.name for index in self.pc.list_indexes()]:
                logger.info(f"Creating new index: {PINECONE_INDEX_NAME}")
                self.pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=EMBEDDING_DIM,
                    metric=PINECONE_METRIC,
                    spec={
                        "serverless": {
                            "cloud": PINECONE_CLOUD,
                            "region": PINECONE_REGION
                        }
                    }
                )
                # Wait for index to be ready
                time.sleep(10)
            
            # Connect to index
            self.index = self.pc.Index(PINECONE_INDEX_NAME)
            
            # Get index stats
            stats = self.index.describe_index_stats()
            logger.info(f"Connected to index. Current vectors: {stats.total_vector_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Pinecone: {e}")
            return False
    
    def clear_index(self) -> bool:
        """Clear existing vectors from index"""
        try:
            if self.index:
                # Get current stats first
                stats = self.index.describe_index_stats()
                current_vectors = stats.total_vector_count
                
                if current_vectors == 0:
                    logger.info("Index is already empty")
                    return True
                
                logger.warning(f"Clearing {current_vectors} vectors from index...")
                
                # Use the correct method to delete all vectors
                # For serverless indexes, we need to delete by namespace or use delete_all
                try:
                    # Try the standard delete_all method first
                    self.index.delete(delete_all=True)
                except Exception as delete_error:
                    logger.warning(f"Standard delete_all failed: {delete_error}")
                    # If that fails, try deleting by namespace
                    if stats.namespaces:
                        for namespace in stats.namespaces.keys():
                            logger.info(f"Deleting namespace: {namespace}")
                            self.index.delete(delete_all=True, namespace=namespace)
                    else:
                        # If no namespaces, try deleting with empty namespace
                        self.index.delete(delete_all=True, namespace="")
                
                # Wait for deletion to complete
                time.sleep(10)
                
                # Verify deletion
                new_stats = self.index.describe_index_stats()
                remaining_vectors = new_stats.total_vector_count
                
                if remaining_vectors == 0:
                    logger.info("Index successfully cleared")
                    return True
                else:
                    logger.warning(f"Index clear incomplete. Remaining vectors: {remaining_vectors}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            return False
    
    def load_chunks(self, file_path: str) -> Generator[Dict[str, Any], None, None]:
        """Load chunks from JSONL file with memory optimization"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        chunk = json.loads(line.strip())
                        if chunk and 'text' in chunk and 'id' in chunk:
                            yield chunk
                        else:
                            logger.warning(f"Invalid chunk at line {line_num} in {file_path}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error at line {line_num} in {file_path}: {e}")
                        continue
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
        except Exception as e:
            logger.error(f"Error loading chunks from {file_path}: {e}")
    
    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts in batches with memory optimization"""
        try:
            # Clear cache before encoding
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Encode with optimizations
            embeddings = self.model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # Important for cosine similarity
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch encoding failed: {e}")
            return np.array([])
    
    def prepare_metadata(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata for Pinecone with size optimization"""
        metadata = chunk.get('metadata', {})
        
        # Optimize metadata for Pinecone (remove large arrays, None values)
        optimized_metadata = {}
        
        # Essential fields for filtering and ranking
        essential_fields = [
            'data_type', 'content_category', 'title', 'language',
            'chunk_type', 'importance_score', 'service_related',
            'file_source', 'department_mentioned', 'location_mentioned'
        ]
        
        for field in essential_fields:
            value = metadata.get(field)
            if value is not None and value != "":
                # Convert boolean to string for Pinecone
                if isinstance(value, bool):
                    optimized_metadata[field] = str(value).lower()
                elif isinstance(value, (int, float)):
                    optimized_metadata[field] = float(value)
                else:
                    optimized_metadata[field] = str(value)[:100]  # Limit string length
        
        # Add keywords (limit to top 5)
        keywords = metadata.get('semantic_keywords', [])
        if keywords:
            optimized_metadata['keywords'] = ','.join(keywords[:5])
        
        # Add source URL if available
        source_url = metadata.get('source_url')
        if source_url:
            optimized_metadata['source_url'] = str(source_url)[:200]
        
        return optimized_metadata
    
    def create_vectors(self, chunks: List[Dict[str, Any]]) -> List[Tuple[str, List[float], Dict[str, Any]]]:
        """Create vector tuples for Pinecone upsert"""
        if not chunks:
            return []
        
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        start_time = time.time()
        embeddings = self.batch_encode(texts)
        embedding_time = time.time() - start_time
        
        if len(embeddings) == 0:
            logger.error("Failed to generate embeddings")
            return []
        
        self.stats['avg_embedding_time'] = embedding_time / len(texts)
        
        # Create vector tuples
        vectors = []
        for i, chunk in enumerate(chunks):
            try:
                vector_id = chunk['id']
                vector_values = embeddings[i].tolist()
                vector_metadata = self.prepare_metadata(chunk)
                
                # Add the original text to metadata (truncated)
                vector_metadata['text'] = chunk['text'][:500]  # Limit text length
                
                vectors.append((vector_id, vector_values, vector_metadata))
                
            except Exception as e:
                logger.error(f"Error creating vector for chunk {chunk.get('id', 'unknown')}: {e}")
                self.stats['failed_upserts'] += 1
                continue
        
        return vectors
    
    def upsert_vectors(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> bool:
        """Upsert vectors to Pinecone with error handling"""
        if not vectors:
            return False
        
        try:
            # Prepare vectors for Pinecone
            vector_data = [
                {
                    "id": vector_id,
                    "values": values,
                    "metadata": metadata
                }
                for vector_id, values, metadata in vectors
            ]
            
            # Upsert to Pinecone
            upsert_response = self.index.upsert(vectors=vector_data)
            
            if upsert_response.upserted_count > 0:
                self.stats['successful_upserts'] += upsert_response.upserted_count
                logger.info(f"Successfully upserted {upsert_response.upserted_count} vectors")
                return True
            else:
                logger.warning("No vectors were upserted")
                return False
                
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            self.stats['failed_upserts'] += len(vectors)
            return False
    
    def process_chunks_file(self, file_path: str) -> bool:
        """Process a single chunks file"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
        
        logger.info(f"Processing file: {os.path.basename(file_path)}")
        
        batch_chunks = []
        total_processed = 0
        
        # Process chunks in batches
        for chunk in self.load_chunks(file_path):
            batch_chunks.append(chunk)
            
            # Process batch when it reaches the limit
            if len(batch_chunks) >= BATCH_SIZE:
                vectors = self.create_vectors(batch_chunks)
                if vectors:
                    self.upsert_vectors(vectors)
                
                total_processed += len(batch_chunks)
                self.stats['total_vectors'] += len(batch_chunks)
                
                # Clear batch and force garbage collection
                batch_chunks = []
                gc.collect()
                
                # Progress update
                if total_processed % (BATCH_SIZE * 10) == 0:
                    logger.info(f"Processed {total_processed} chunks from {os.path.basename(file_path)}")
        
        # Process remaining chunks
        if batch_chunks:
            vectors = self.create_vectors(batch_chunks)
            if vectors:
                self.upsert_vectors(vectors)
            
            total_processed += len(batch_chunks)
            self.stats['total_vectors'] += len(batch_chunks)
        
        logger.info(f"Completed processing {file_path}. Total chunks: {total_processed}")
        return True
    
    def run_pipeline(self, clear_existing: bool = True) -> bool:
        """Run the complete embedding and upsert pipeline"""
        start_time = time.time()
        
        logger.info("Starting advanced embedding and upsert pipeline...")
        
        # Setup components
        if not self.setup_model():
            return False
        
        if not self.setup_pinecone():
            return False
        
        # Clear existing vectors if requested
        if clear_existing:
            if not self.clear_index():
                logger.error("Failed to clear index! This means you may be mixing old and new data.")
                logger.error("Please either:")
                logger.error("1. Manually delete the index in Pinecone console and rerun")
                logger.error("2. Set clear_existing=False to append to existing data")
                logger.error("3. Use a different index name")
                return False
        
        # Process all chunk files
        chunk_files = [
            'all_chunks.jsonl',  # Process the comprehensive file
            # 'web_content_chunks.jsonl',
            # 'tabular_data_chunks.jsonl', 
            # 'service_chunks.jsonl'
        ]
        
        success_count = 0
        for filename in chunk_files:
            file_path = os.path.join(DATA_DIR, filename)
            if self.process_chunks_file(file_path):
                success_count += 1
        
        # Calculate final stats
        self.stats['processing_time'] = time.time() - start_time
        
        # Verify index stats
        try:
            final_stats = self.index.describe_index_stats()
            logger.info(f"Final index stats: {final_stats.total_vector_count} vectors")
        except:
            pass
        
        # Generate report
        self.generate_report()
        
        return success_count > 0
    
    def generate_report(self) -> None:
        """Generate processing report"""
        report = {
            "pipeline_completed_at": datetime.utcnow().isoformat() + "Z",
            "model_used": self.model_name,
            "embedding_dimension": EMBEDDING_DIM,
            "processing_stats": self.stats,
            "performance_metrics": {
                "vectors_per_second": round(
                    self.stats['total_vectors'] / max(1, self.stats['processing_time']), 2
                ),
                "success_rate": round(
                    (self.stats['successful_upserts'] / max(1, self.stats['total_vectors'])) * 100, 2
                ),
                "avg_embedding_time_ms": round(self.stats['avg_embedding_time'] * 1000, 2)
            },
            "hardware_config": {
                "device": DEVICE,
                "batch_size": BATCH_SIZE,
                "max_memory_usage": MAX_MEMORY_USAGE
            }
        }
        
        # Save report
        report_path = os.path.join(DATA_DIR, 'embedding_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info("=" * 60)
        logger.info("EMBEDDING PIPELINE COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total vectors processed: {self.stats['total_vectors']}")
        logger.info(f"Successful upserts: {self.stats['successful_upserts']}")
        logger.info(f"Failed upserts: {self.stats['failed_upserts']}")
        logger.info(f"Success rate: {report['performance_metrics']['success_rate']}%")
        logger.info(f"Processing time: {self.stats['processing_time']:.2f} seconds")
        logger.info(f"Vectors per second: {report['performance_metrics']['vectors_per_second']}")
        logger.info("=" * 60)
        
        print(json.dumps(report, ensure_ascii=False, indent=2))

def main():
    """Main execution function"""
    pipeline = AdvancedEmbeddingPipeline()
    
    # Check for required environment variables
    if not os.getenv('PINECONE_API_KEY'):
        logger.error("Please set PINECONE_API_KEY environment variable")
        return False
    
    # Run the pipeline
    success = pipeline.run_pipeline(clear_existing=True)
    
    if success:
        logger.info("Pipeline completed successfully!")
        return True
    else:
        logger.error("Pipeline failed!")
        return False

if __name__ == "__main__":
    main()
