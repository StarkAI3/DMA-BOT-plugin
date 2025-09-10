#!/usr/bin/env python3
"""
Quick Setup and Verification Script for DMA RAG System
Verifies Stella model and guides through API key setup
"""

import os
import sys
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

def check_embedding_model():
    """Check if E5-base-v2 embedding model is available and functional"""
    print("🔍 Checking E5-base-v2 embedding model...")
    
    # E5 model will be downloaded from HuggingFace if needed
    
    try:
        print("⏳ Loading E5-base-v2 model (this may take a moment)...")
        
        model = SentenceTransformer('intfloat/e5-base-v2', device='cpu')
        
        # Test encoding
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)
        
        print(f"✅ E5-base-v2 model loaded successfully!")
        print(f"   Model: intfloat/e5-base-v2")
        print(f"   Embedding dimension: {len(embedding)} (768-dim)")
        print(f"   Test embedding shape: {embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load E5-base-v2 model: {e}")
        return False

def check_api_keys():
    """Check API keys and provide setup instructions"""
    print("\n🔑 Checking API keys...")
    
    pinecone_key = os.getenv('PINECONE_API_KEY')
    gemini_key = os.getenv('GEMINI_API_KEY')
    
    if pinecone_key:
        print("✅ PINECONE_API_KEY is set")
    else:
        print("❌ PINECONE_API_KEY not set")
        print("   Please run: export PINECONE_API_KEY='your_pinecone_api_key_here'")
    
    if gemini_key:
        print("✅ GEMINI_API_KEY is set")
    else:
        print("❌ GEMINI_API_KEY not set")
        print("   Please run: export GEMINI_API_KEY='your_gemini_api_key_here'")
    
    return bool(pinecone_key and gemini_key)

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'sentence_transformers',
        'pinecone',
        'google.generativeai',
        'nltk',
        'numpy',
        'tqdm',
        'flask'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'google.generativeai':
                import google.generativeai
            elif package == 'pinecone':
                import pinecone
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n🚨 Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_virtual_environment():
    """Check if virtual environment is activated"""
    print("\n🐍 Checking Python environment...")
    
    python_path = sys.executable
    print(f"Python executable: {python_path}")
    
    if 'venv' in python_path:
        print("✅ Virtual environment is activated")
        return True
    else:
        print("⚠️  Virtual environment might not be activated")
        print("   Consider running: source venv/bin/activate")
        return False

def provide_next_steps(model_ok, keys_ok, deps_ok, venv_ok):
    """Provide next steps based on checks"""
    print("\n" + "="*60)
    print("SETUP STATUS SUMMARY")
    print("="*60)
    
    print(f"E5-base-v2 Model: {'✅ Ready' if model_ok else '❌ Issue'}")
    print(f"API Keys:         {'✅ Set' if keys_ok else '❌ Missing'}")
    print(f"Dependencies:     {'✅ Installed' if deps_ok else '❌ Missing'}")
    print(f"Virtual Env:      {'✅ Active' if venv_ok else '⚠️  Check'}")
    
    print("\n📋 NEXT STEPS:")
    
    if not venv_ok:
        print("1. Activate virtual environment:")
        print("   source venv/bin/activate")
    
    if not deps_ok:
        print("2. Install missing dependencies:")
        print("   pip install sentence-transformers pinecone-client google-generativeai nltk numpy tqdm flask flask-cors")
    
    if not keys_ok:
        print("3. Set API keys:")
        print("   export PINECONE_API_KEY='your_pinecone_api_key_here'")
        print("   export GEMINI_API_KEY='your_gemini_api_key_here'")
    
    if model_ok and keys_ok and deps_ok:
        print("🎉 ALL CHECKS PASSED! You can now run the system:")
        print()
        print("Option 1 - Automated rebuild:")
        print("   python rebuild_rag_system.py")
        print()
        print("Option 2 - Manual step-by-step:")
        print("   python src/advanced_preprocess.py")
        print("   python src/advanced_embed_upsert.py")
        print("   python src/server.py")
        print()
        print("Option 3 - Test query system:")
        print("   python src/advanced_query_rag.py")
    else:
        print("\n⚠️  Please address the issues above before proceeding.")

def main():
    """Main verification function"""
    print("🚀 DMA RAG System - Quick Setup Verification")
    print("Using intfloat/e5-base-v2 embedding model (768-dim)")
    print("="*60)
    
    # Run all checks
    model_ok = check_embedding_model()
    venv_ok = check_virtual_environment()
    deps_ok = check_dependencies()
    keys_ok = check_api_keys()
    
    # Provide guidance
    provide_next_steps(model_ok, keys_ok, deps_ok, venv_ok)

if __name__ == "__main__":
    main()
