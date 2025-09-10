# 🚀 DMA RAG System Setup & Usage Guide

## Overview
This guide will help you set up and run the enhanced DMA RAG system using **only** the `dunzhang/stella_en_400M_v5` embedding model for maximum accuracy.

## Prerequisites Verification

### 1. Check Stella Model Availability
```bash
ls -la models/models--dunzhang--stella_en_400M_v5/
```
✅ **Confirmed**: Your Stella model is available at the correct location.

### 2. Environment Setup

#### Required API Keys
```bash
# Set your Pinecone API key
export PINECONE_API_KEY="your_pinecone_api_key_here"

# Set your Gemini API key
export GEMINI_API_KEY="your_gemini_api_key_here"

# Optional: Add to ~/.bashrc for persistence
echo 'export PINECONE_API_KEY="your_pinecone_api_key_here"' >> ~/.bashrc
echo 'export GEMINI_API_KEY="your_gemini_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

#### Activate Virtual Environment
```bash
source venv/bin/activate
```

## Installation & Setup

### Step 1: Install Dependencies
```bash
pip install --upgrade sentence-transformers pinecone-client google-generativeai nltk numpy tqdm flask flask-cors
```

### Step 2: Run the Complete Rebuild
```bash
# Make sure you're in the project directory
cd /home/stark/Desktop/DMA_BOT

# Run the automated rebuild script
python rebuild_rag_system.py
```

The script will:
1. ✅ Verify environment and API keys
2. ✅ Install/update dependencies
3. ✅ Process your data with advanced chunking
4. ✅ Generate embeddings using Stella model only
5. ✅ Upload to Pinecone vector database
6. ✅ Test the query system
7. ✅ Update web interface

## Manual Step-by-Step Process (Alternative)

If you prefer to run each step manually:

### Step 1: Advanced Data Preprocessing
```bash
python src/advanced_preprocess.py
```
**Output**: Optimized chunks in `optimized_data/` directory

### Step 2: Generate Embeddings & Upload to Pinecone
```bash
python src/advanced_embed_upsert.py
```
**Output**: Vector database populated with Stella embeddings

### Step 3: Test Query System
```bash
python src/advanced_query_rag.py
```
**Interactive testing interface**

## Running the System

### Option 1: Web Interface (Recommended)
```bash
# Start the web server
python src/server.py
```

Then open your browser to: **http://localhost:5000**

### Option 2: Command Line Interface
```bash
# Interactive query mode
python src/advanced_query_rag.py
```

## System Configuration

### Stella Model Configuration
The system is now configured to use **ONLY** the Stella model:
- **Model Path**: `models/models--dunzhang--stella_en_400M_v5`
- **Embedding Dimension**: 1024
- **Device**: CPU (optimized for Intel i3)
- **Batch Size**: 8 (optimized for 12GB RAM)

### Key Features
- ✅ **Semantic Chunking**: Respects sentence boundaries
- ✅ **Intent Detection**: Understands user query intent
- ✅ **Smart Filtering**: Uses metadata for precise retrieval
- ✅ **Gemini 2.0 Flash**: High-quality response generation
- ✅ **Hardware Optimized**: Tuned for your system specs

## Testing the System

### Sample Queries to Test
```
1. "How to apply for water connection?"
2. "What documents are required for trade license?"
3. "Contact information for DMA office"
4. "Property tax payment procedure"
5. "Marriage registration process and fees"
```

### Expected Response Quality
- **Accurate Information**: Based on your DMA data
- **Step-by-step Procedures**: For application processes
- **Contact Details**: When available in source data
- **Source Attribution**: Links to original documents
- **Fast Response**: < 3 seconds per query

## Troubleshooting

### Common Issues

#### 1. Stella Model Not Found
```bash
# Verify model exists
ls -la models/models--dunzhang--stella_en_400M_v5/

# If missing, the model should already be downloaded
# Contact support if the model directory is empty
```

#### 2. API Key Issues
```bash
# Check if keys are set
echo $PINECONE_API_KEY
echo $GEMINI_API_KEY

# Set keys if missing
export PINECONE_API_KEY="your_key"
export GEMINI_API_KEY="your_key"
```

#### 3. Memory Issues
```bash
# Monitor memory usage
htop

# If needed, reduce batch size in advanced_embed_upsert.py
# BATCH_SIZE = 4  # Reduce from 8 to 4
```

#### 4. Pinecone Connection Issues
```bash
# Test Pinecone connection
python -c "
from pinecone import Pinecone
import os
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
print('Pinecone connected successfully!')
print(f'Available indexes: {[index.name for index in pc.list_indexes()]}')
"
```

### Log Locations
- **Preprocessing**: Console output during `advanced_preprocess.py`
- **Embedding**: Console output during `advanced_embed_upsert.py`
- **Query System**: Console output during server startup
- **Reports**: Check `optimized_data/` for processing reports

## Performance Optimization

### For Better Speed
```python
# In advanced_embed_upsert.py, you can adjust:
BATCH_SIZE = 6  # Increase if you have more RAM
```

### For Better Accuracy
```python
# In advanced_query_rag.py, you can adjust:
TOP_K_RETRIEVAL = 30  # Retrieve more candidates
FINAL_CONTEXT_LIMIT = 10  # Use more context
```

## System Monitoring

### Check System Health
```bash
curl http://localhost:5000/health
```

### Monitor Vector Database
```python
from pinecone import Pinecone
import os

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index("dma-knowledge-base-v2")
stats = index.describe_index_stats()
print(f"Total vectors: {stats.total_vector_count}")
```

## Next Steps

1. **Test Extensively**: Try various query types
2. **Monitor Performance**: Check response times and accuracy
3. **Gather Feedback**: Test with real users
4. **Optimize Further**: Adjust parameters based on results

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review console logs for error messages
3. Verify all prerequisites are met
4. Test individual components separately

---

**Your RAG system is now optimized for maximum accuracy using only the Stella embedding model!** 🎉
