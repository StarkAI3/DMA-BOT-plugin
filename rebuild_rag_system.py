#!/usr/bin/env python3
"""
Complete RAG System Rebuild Script
Execute this to rebuild the entire system with high accuracy optimizations
"""

import os
import sys
import subprocess
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """Check required environment variables and dependencies"""
    logger.info("Checking environment...")
    
    required_vars = ['PINECONE_API_KEY', 'GEMINI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables before running the script:")
        for var in missing_vars:
            logger.error(f"export {var}='your_api_key_here'")
        return False
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not sys.base_prefix != sys.prefix:
        logger.warning("Virtual environment might not be activated")
        logger.warning("Consider running: source venv/bin/activate")
    
    logger.info("Environment check passed!")
    return True

def install_dependencies():
    """Install required dependencies"""
    logger.info("Installing/updating dependencies...")
    
    dependencies = [
        "sentence-transformers",
        "pinecone-client",
        "google-generativeai",
        "nltk",
        "numpy",
        "tqdm"
    ]
    
    try:
        for dep in dependencies:
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", dep], 
                         check=True, capture_output=True, text=True)
        logger.info("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def run_preprocessing():
    """Run advanced preprocessing"""
    logger.info("="*60)
    logger.info("STEP 1: ADVANCED PREPROCESSING")
    logger.info("="*60)
    
    try:
        from advanced_preprocess import AdvancedPreprocessor
        
        preprocessor = AdvancedPreprocessor()
        preprocessor.run()
        
        logger.info("Advanced preprocessing completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return False

def run_embedding_upsert():
    """Run advanced embedding and upsert"""
    logger.info("="*60)
    logger.info("STEP 2: ADVANCED EMBEDDING & VECTOR DB UPSERT")
    logger.info("="*60)
    
    try:
        from advanced_embed_upsert import AdvancedEmbeddingPipeline
        
        pipeline = AdvancedEmbeddingPipeline()
        success = pipeline.run_pipeline(clear_existing=True)
        
        if success:
            logger.info("Advanced embedding and upsert completed successfully!")
            return True
        else:
            logger.error("Embedding and upsert failed!")
            return False
        
    except Exception as e:
        logger.error(f"Embedding and upsert failed: {e}")
        return False

def test_query_system():
    """Test the query system"""
    logger.info("="*60)
    logger.info("STEP 3: TESTING QUERY SYSTEM")
    logger.info("="*60)
    
    try:
        from advanced_query_rag import AdvancedRAGQuerySystem
        
        rag = AdvancedRAGQuerySystem()
        
        if not rag.initialize():
            logger.error("Failed to initialize RAG system")
            return False
        
        # Test queries
        test_queries = [
            "How to apply for water connection?",
            "What documents are required for trade license?",
            "Contact information for DMA office",
            "Property tax payment procedure"
        ]
        
        logger.info("Running test queries...")
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"Test {i}: {query}")
            
            try:
                result = rag.query(query)
                
                if result["response"] and "error" not in result["response"].lower():
                    logger.info(f"✅ Test {i} passed - Response generated successfully")
                    logger.info(f"   Sources found: {result['metadata']['results_found']}")
                    logger.info(f"   Processing time: {result['processing_time']:.2f}s")
                else:
                    logger.warning(f"⚠️  Test {i} generated error response")
                
            except Exception as e:
                logger.error(f"❌ Test {i} failed: {e}")
                return False
            
            time.sleep(1)  # Small delay between tests
        
        logger.info("All test queries completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Query system test failed: {e}")
        return False

def update_server():
    """Update the server to use the new RAG system"""
    logger.info("="*60)
    logger.info("STEP 4: UPDATING SERVER")
    logger.info("="*60)
    
    server_code = '''#!/usr/bin/env python3
"""
Enhanced DMA RAG Server with Advanced Query System
"""

import os
import sys
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from advanced_query_rag import AdvancedRAGQuerySystem

app = Flask(__name__, static_folder='static', template_folder='static')
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG system instance
rag_system = None

def initialize_rag():
    """Initialize the RAG system"""
    global rag_system
    
    try:
        rag_system = AdvancedRAGQuerySystem()
        if rag_system.initialize():
            logger.info("RAG system initialized successfully!")
            return True
        else:
            logger.error("Failed to initialize RAG system")
            return False
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        return False

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    """Handle query requests"""
    global rag_system
    
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        
        if not user_query:
            return jsonify({
                'success': False,
                'error': 'Please provide a valid query'
            }), 400
        
        if not rag_system:
            return jsonify({
                'success': False,
                'error': 'RAG system not initialized'
            }), 500
        
        # Process the query
        result = rag_system.query(user_query)
        
        return jsonify({
            'success': True,
            'query': result['query'],
            'response': result['response'],
            'sources': result['sources'],
            'metadata': {
                'processing_time': round(result['processing_time'], 2),
                'results_found': result['metadata']['results_found'],
                'intent': result['metadata']['intent']
            }
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while processing your query'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    global rag_system
    
    status = {
        'status': 'healthy' if rag_system else 'unhealthy',
        'rag_initialized': rag_system is not None,
        'timestamp': os.popen('date').read().strip()
    }
    
    return jsonify(status)

if __name__ == '__main__':
    logger.info("Starting DMA RAG Server...")
    
    # Initialize RAG system
    if initialize_rag():
        logger.info("Server starting on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("Failed to start server - RAG system initialization failed")
        sys.exit(1)
'''
    
    try:
        with open('src/server.py', 'w', encoding='utf-8') as f:
            f.write(server_code)
        
        logger.info("Server updated successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update server: {e}")
        return False

def update_frontend():
    """Update the frontend interface"""
    logger.info("Updating frontend interface...")
    
    enhanced_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DMA RAG Chatbot - Enhanced</title>
    <link rel="icon" href="seal.svg" type="image/svg+xml">
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <header class="header">
            <img src="seal.svg" alt="DMA Seal" class="logo">
            <div class="header-text">
                <h1>DMA RAG Assistant</h1>
                <p>Directorate of Municipal Administration, Maharashtra</p>
                <div class="status-indicator" id="status">
                    <span class="status-dot"></span>
                    <span class="status-text">Initializing...</span>
                </div>
            </div>
        </header>

        <div class="chat-container">
            <div class="chat-messages" id="chat-messages">
                <div class="message assistant-message">
                    <div class="message-content">
                        <strong>🚀 Enhanced DMA RAG Assistant Ready!</strong><br>
                        Ask me about municipal services, applications, procedures, and more.<br>
                        <em>New features: Advanced semantic search, better context understanding, and improved accuracy!</em>
                    </div>
                </div>
            </div>

            <div class="input-container">
                <div class="input-wrapper">
                    <input type="text" id="user-input" placeholder="Ask about municipal services, applications, procedures..." maxlength="500">
                    <button id="send-button" onclick="sendMessage()">
                        <span class="send-text">Send</span>
                        <span class="send-loading" style="display: none;">Processing...</span>
                    </button>
                </div>
                <div class="suggestions">
                    <button class="suggestion-btn" onclick="setSuggestion('How to apply for water connection?')">Water Connection</button>
                    <button class="suggestion-btn" onclick="setSuggestion('Trade license application procedure')">Trade License</button>
                    <button class="suggestion-btn" onclick="setSuggestion('Property tax payment process')">Property Tax</button>
                    <button class="suggestion-btn" onclick="setSuggestion('Marriage registration documents required')">Marriage Registration</button>
                </div>
            </div>
        </div>
    </div>

    <script src="main.js"></script>
</body>
</html>'''
    
    enhanced_js = '''// Enhanced DMA RAG Chatbot JavaScript
let isProcessing = false;

// Check system status on load
document.addEventListener('DOMContentLoaded', function() {
    checkSystemStatus();
    
    // Setup enter key listener
    document.getElementById('user-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !isProcessing) {
            sendMessage();
        }
    });
});

async function checkSystemStatus() {
    try {
        const response = await fetch('/health');
        const status = await response.json();
        
        const statusElement = document.getElementById('status');
        const dot = statusElement.querySelector('.status-dot');
        const text = statusElement.querySelector('.status-text');
        
        if (status.rag_initialized) {
            dot.className = 'status-dot online';
            text.textContent = 'System Ready';
        } else {
            dot.className = 'status-dot offline';
            text.textContent = 'System Offline';
        }
    } catch (error) {
        console.error('Status check failed:', error);
        const statusElement = document.getElementById('status');
        const dot = statusElement.querySelector('.status-dot');
        const text = statusElement.querySelector('.status-text');
        
        dot.className = 'status-dot offline';
        text.textContent = 'Connection Error';
    }
}

function setSuggestion(suggestion) {
    document.getElementById('user-input').value = suggestion;
    document.getElementById('user-input').focus();
}

async function sendMessage() {
    if (isProcessing) return;
    
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const query = userInput.value.trim();
    
    if (!query) return;
    
    // Set processing state
    isProcessing = true;
    userInput.disabled = true;
    sendButton.disabled = true;
    sendButton.querySelector('.send-text').style.display = 'none';
    sendButton.querySelector('.send-loading').style.display = 'inline';
    
    // Add user message to chat
    addMessage(query, 'user');
    userInput.value = '';
    
    // Add thinking indicator
    const thinkingId = addThinkingIndicator();
    
    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query })
        });
        
        const data = await response.json();
        
        // Remove thinking indicator
        removeThinkingIndicator(thinkingId);
        
        if (data.success) {
            // Add assistant response
            addMessage(data.response, 'assistant', data.sources, data.metadata);
        } else {
            addMessage('Sorry, I encountered an error: ' + (data.error || 'Unknown error'), 'assistant', [], null, true);
        }
        
    } catch (error) {
        removeThinkingIndicator(thinkingId);
        addMessage('Sorry, I couldn\\'t process your request. Please check your connection and try again.', 'assistant', [], null, true);
        console.error('Query failed:', error);
    } finally {
        // Reset processing state
        isProcessing = false;
        userInput.disabled = false;
        sendButton.disabled = false;
        sendButton.querySelector('.send-text').style.display = 'inline';
        sendButton.querySelector('.send-loading').style.display = 'none';
        userInput.focus();
    }
}

function addMessage(content, sender, sources = [], metadata = null, isError = false) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message${isError ? ' error-message' : ''}`;
    
    let messageHTML = `<div class="message-content">${formatMessage(content)}</div>`;
    
    // Add sources if available
    if (sources && sources.length > 0) {
        messageHTML += '<div class="sources"><strong>📚 Sources:</strong>';
        sources.forEach((source, index) => {
            const sourceHTML = `
                <div class="source-item">
                    <span class="source-title">${source.title}</span>
                    ${source.url ? `<a href="${source.url}" target="_blank" class="source-link">🔗 Link</a>` : ''}
                    <span class="source-relevance">Relevance: ${(source.relevance_score * 100).toFixed(0)}%</span>
                </div>
            `;
            messageHTML += sourceHTML;
        });
        messageHTML += '</div>';
    }
    
    // Add metadata if available
    if (metadata) {
        messageHTML += `
            <div class="metadata">
                <small>
                    ⏱️ ${metadata.processing_time}s | 
                    📊 ${metadata.results_found} results | 
                    🎯 Intent: ${metadata.intent.intents.join(', ')}
                </small>
            </div>
        `;
    }
    
    messageDiv.innerHTML = messageHTML;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addThinkingIndicator() {
    const chatMessages = document.getElementById('chat-messages');
    const thinkingDiv = document.createElement('div');
    const thinkingId = 'thinking-' + Date.now();
    thinkingDiv.id = thinkingId;
    thinkingDiv.className = 'message assistant-message thinking';
    thinkingDiv.innerHTML = `
        <div class="message-content">
            <span class="thinking-dots">🤔 Analyzing your query</span>
        </div>
    `;
    
    chatMessages.appendChild(thinkingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return thinkingId;
}

function removeThinkingIndicator(thinkingId) {
    const thinkingElement = document.getElementById(thinkingId);
    if (thinkingElement) {
        thinkingElement.remove();
    }
}

function formatMessage(message) {
    // Convert URLs to clickable links
    const urlRegex = /(https?:\\/\\/[^\\s]+)/g;
    message = message.replace(urlRegex, '<a href="$1" target="_blank">$1</a>');
    
    // Convert line breaks to HTML
    message = message.replace(/\\n/g, '<br>');
    
    // Format numbered lists
    message = message.replace(/^(\\d+\\..+)$/gm, '<li>$1</li>');
    
    return message;
}

// Periodic status check
setInterval(checkSystemStatus, 30000); // Check every 30 seconds'''
    
    enhanced_css = '''/* Enhanced DMA RAG Chatbot Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
}

.container {
    background: white;
    border-radius: 20px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
    width: 95%;
    max-width: 900px;
    max-height: 90vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.header {
    background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    color: white;
    padding: 20px;
    display: flex;
    align-items: center;
    gap: 15px;
}

.logo {
    width: 60px;
    height: 60px;
    filter: brightness(0) invert(1);
}

.header-text h1 {
    font-size: 24px;
    font-weight: 600;
    margin-bottom: 5px;
}

.header-text p {
    font-size: 14px;
    opacity: 0.9;
    margin-bottom: 8px;
}

.status-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #95a5a6;
    animation: pulse 2s infinite;
}

.status-dot.online {
    background: #2ecc71;
}

.status-dot.offline {
    background: #e74c3c;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.chat-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    background: #f8f9fa;
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.message {
    max-width: 80%;
    animation: fadeInUp 0.3s ease-out;
}

.user-message {
    align-self: flex-end;
}

.assistant-message {
    align-self: flex-start;
}

.message-content {
    padding: 15px 20px;
    border-radius: 18px;
    line-height: 1.5;
    word-wrap: break-word;
}

.user-message .message-content {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-bottom-right-radius: 5px;
}

.assistant-message .message-content {
    background: white;
    color: #2c3e50;
    border: 1px solid #e1e8ed;
    border-bottom-left-radius: 5px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

.error-message .message-content {
    background: #fee;
    border-color: #fcc;
    color: #c33;
}

.thinking .message-content {
    background: #f0f0f0;
    color: #666;
    font-style: italic;
}

.sources {
    margin-top: 12px;
    padding: 12px;
    background: #f8f9fa;
    border-radius: 8px;
    border-left: 3px solid #667eea;
    font-size: 13px;
}

.source-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
    border-bottom: 1px solid #e1e8ed;
}

.source-item:last-child {
    border-bottom: none;
}

.source-title {
    flex: 1;
    font-weight: 500;
    color: #2c3e50;
}

.source-link {
    color: #667eea;
    text-decoration: none;
    font-size: 12px;
}

.source-link:hover {
    text-decoration: underline;
}

.source-relevance {
    font-size: 11px;
    color: #666;
    background: #e9ecef;
    padding: 2px 6px;
    border-radius: 10px;
}

.metadata {
    margin-top: 8px;
    color: #666;
    font-size: 11px;
}

.input-container {
    padding: 20px;
    background: white;
    border-top: 1px solid #e1e8ed;
}

.input-wrapper {
    display: flex;
    gap: 10px;
    margin-bottom: 12px;
}

#user-input {
    flex: 1;
    padding: 15px 20px;
    border: 2px solid #e1e8ed;
    border-radius: 25px;
    font-size: 14px;
    outline: none;
    transition: border-color 0.3s ease;
}

#user-input:focus {
    border-color: #667eea;
}

#user-input:disabled {
    background: #f8f9fa;
    cursor: not-allowed;
}

#send-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 15px 25px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
    min-width: 100px;
}

#send-button:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

#send-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
}

.suggestions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.suggestion-btn {
    background: #f8f9fa;
    border: 1px solid #e1e8ed;
    border-radius: 15px;
    padding: 8px 12px;
    font-size: 12px;
    color: #495057;
    cursor: pointer;
    transition: all 0.3s ease;
}

.suggestion-btn:hover {
    background: #667eea;
    color: white;
    border-color: #667eea;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Responsive design */
@media (max-width: 768px) {
    .container {
        width: 100%;
        height: 100vh;
        border-radius: 0;
        max-height: none;
    }
    
    .header {
        padding: 15px;
    }
    
    .logo {
        width: 40px;
        height: 40px;
    }
    
    .header-text h1 {
        font-size: 20px;
    }
    
    .message {
        max-width: 90%;
    }
    
    .suggestions {
        gap: 6px;
    }
    
    .suggestion-btn {
        font-size: 11px;
        padding: 6px 10px;
    }
}'''
    
    try:
        # Update HTML
        with open('static/index.html', 'w', encoding='utf-8') as f:
            f.write(enhanced_html)
        
        # Update JavaScript
        with open('static/main.js', 'w', encoding='utf-8') as f:
            f.write(enhanced_js)
        
        # Update CSS
        with open('static/styles.css', 'w', encoding='utf-8') as f:
            f.write(enhanced_css)
        
        logger.info("Frontend updated successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update frontend: {e}")
        return False

def main():
    """Main rebuild execution"""
    print("🚀 DMA RAG System Rebuild Script")
    print("=" * 60)
    print("This script will rebuild your RAG system with high accuracy optimizations:")
    print("1. Advanced semantic chunking")
    print("2. Enhanced metadata enrichment") 
    print("3. Optimized embedding pipeline")
    print("4. Intelligent query processing")
    print("5. Gemini 2.0 Flash integration")
    print("=" * 60)
    
    # Confirmation
    response = input("Do you want to proceed? This will clear your existing vector database. (y/N): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    start_time = time.time()
    
    # Step-by-step execution
    steps = [
        ("Environment Check", check_environment),
        ("Dependencies Installation", install_dependencies),
        ("Advanced Preprocessing", run_preprocessing),
        ("Embedding & Vector DB", run_embedding_upsert),
        ("Query System Test", test_query_system),
        ("Server Update", update_server),
        ("Frontend Update", update_frontend)
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        logger.info(f"Executing: {step_name}")
        
        try:
            if step_func():
                logger.info(f"✅ {step_name} completed successfully!")
            else:
                logger.error(f"❌ {step_name} failed!")
                failed_steps.append(step_name)
                
                # Ask if user wants to continue
                if step_name in ["Environment Check", "Dependencies Installation"]:
                    logger.error("Critical step failed. Cannot continue.")
                    return
                    
                response = input(f"Continue despite {step_name} failure? (y/N): ")
                if response.lower() != 'y':
                    return
                    
        except Exception as e:
            logger.error(f"❌ {step_name} failed with exception: {e}")
            failed_steps.append(step_name)
            
        time.sleep(1)
    
    # Final summary
    total_time = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info("REBUILD COMPLETION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    
    if failed_steps:
        logger.warning(f"Failed steps: {', '.join(failed_steps)}")
        logger.warning("Some components may not work correctly.")
    else:
        logger.info("🎉 All steps completed successfully!")
        logger.info("Your RAG system has been rebuilt with high accuracy optimizations!")
        
    logger.info("\nNext steps:")
    logger.info("1. Start the server: python src/server.py")
    logger.info("2. Test the system at: http://localhost:5000")
    logger.info("3. Monitor logs for any issues")
    
    print("\n" + "=" * 60)
    print("RAG System Rebuild Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
