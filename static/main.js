const form = document.getElementById('chat-form');
const messagesEl = document.getElementById('messages');
const input = document.getElementById('query');
const inputContainer = document.getElementById('inputContainer');
const welcomeBlock = document.getElementById('welcomeBlock');
const startChatBtn = document.getElementById('startChatBtn');
const sendBtn = document.getElementById('sendBtn');

// Global error handler to catch any JavaScript errors
window.addEventListener('error', function(e) {
  console.error('JavaScript error:', e.error);
});

// Generate a simple session ID that works in all browsers
function generateSessionId() {
  return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

let sessionId = generateSessionId();
let chatStarted = false;

// Add typing indicator functionality
let typingTimeout;

function showTypingIndicator() {
  const typingEl = document.createElement('div');
  typingEl.className = 'msg bot typing-indicator';
  typingEl.innerHTML = `
    <div class="typing-dots">
      <span></span>
      <span></span>
      <span></span>
    </div>
  `;
  messagesEl.appendChild(typingEl);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return typingEl;
}

function hideTypingIndicator(typingEl) {
  if (typingEl && typingEl.parentNode) {
    typingEl.remove();
  }
}

function formatMessage(text) {
  text = text.replace(/(https?:\/\/[^\s)]+)([).,]?)/g, '<a href="$1" target="_blank">$1</a>$2');
  return text.replace(/\n/g, "<br>");
}

function appendMessage(text, who = 'bot', sources = [], language = null) {
  const wrap = document.createElement('div');
  wrap.className = `msg ${who}`;
  
  // Add timestamp
  const timestamp = new Date().toLocaleTimeString('en-US', { 
    hour: '2-digit', 
    minute: '2-digit',
    hour12: false 
  });
  
  // Format message content
  if (who === 'bot') {
    let messageContent = `<div class="message-text">${formatMessage(text)}</div>`;
    
    // Add language indicator for bot messages if language is detected
    if (language) {
      const languageText = language === 'mr' ? 'मराठी' : 'English';
      messageContent = `<div class="language-indicator ${language}">${languageText}</div>` + messageContent;
    }
    
    wrap.innerHTML = messageContent + 
      `<div class="message-time">${timestamp}</div>`;
  } else {
    wrap.innerHTML = `
      <div class="message-text">${text}</div>
      <div class="message-time">${timestamp}</div>
    `;
  }
  
  messagesEl.appendChild(wrap);
  
  // Add sources if available
  if (sources && sources.length > 0) {
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = 'Sources:';
    
    const list = document.createElement('div');
    list.className = 'sources';
    
    sources.forEach(s => {
      if (!s.url) return;
      const a = document.createElement('a');
      a.className = 'source';
      a.href = s.url;
      a.target = '_blank';
      a.rel = 'noreferrer';
      a.textContent = s.title || new URL(s.url).hostname;
      list.appendChild(a);
    });
    
    const box = document.createElement('div');
    box.appendChild(meta);
    box.appendChild(list);
    messagesEl.appendChild(box);
  }
  
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// Start chat functionality
startChatBtn.onclick = function () {
  welcomeBlock.style.display = "none";
  inputContainer.style.display = "flex";
  chatStarted = true;
  input.focus();
};

// Add a fallback event listener using addEventListener
if (startChatBtn) {
  startChatBtn.addEventListener('click', function(e) {
    e.preventDefault();
    e.stopPropagation();
    
    if (welcomeBlock && inputContainer) {
      welcomeBlock.style.display = "none";
      inputContainer.style.display = "flex";
      chatStarted = true;
      input.focus();
    }
  });
}

// Send message functionality
async function sendMessage() {
  const question = input.value.trim();
  if (!question) return;
  
  appendMessage(question, 'user');
  input.value = '';
  
  // Show typing indicator
  const typingEl = showTypingIndicator();
  
  // Clear any existing timeout
  if (typingTimeout) {
    clearTimeout(typingTimeout);
  }
  
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        query: question, 
        session_id: sessionId,
        top_k: 5 
      }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    // Hide typing indicator
    hideTypingIndicator(typingEl);
    
    // Add bot response with language detection
    appendMessage(data.answer, 'bot', data.sources || [], data.detected_lang);
    
  } catch (error) {
    console.error('Error sending message:', error);
    
    // Hide typing indicator
    hideTypingIndicator(typingEl);
    
    // Show error message
    appendMessage('Sorry, there was an error processing your request. Please try again.', 'bot');
  }
}

// Event listeners
sendBtn.onclick = sendMessage;

input.addEventListener('keydown', e => {
  if (e.key === 'Enter') {
    e.preventDefault();
    sendMessage();
  }
});

// Add input focus and enter key handling
input.addEventListener('keypress', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Add input focus for better UX
input.addEventListener('focus', () => {
  input.style.borderColor = 'var(--dma-dark-blue)';
});

input.addEventListener('blur', () => {
  input.style.borderColor = '#ccc';
});

// Auto-scroll to bottom when new messages are added
const observer = new MutationObserver(() => {
  messagesEl.scrollTop = messagesEl.scrollHeight;
});

observer.observe(messagesEl, {
  childList: true,
  subtree: true
});

// Add smooth scrolling
messagesEl.style.scrollBehavior = 'smooth';

// Keyboard shortcuts
document.addEventListener("keydown", (e) => {
  if (!chatStarted && (e.key === "Enter" || e.key === " ")) {
    e.preventDefault();
    startChatBtn.click();
  }
});

// Add CSS for typing indicator
const style = document.createElement('style');
style.textContent = `
  .typing-indicator {
    background: #fbeaea !important;
    border-left: 4px solid var(--dma-dark-blue) !important;
    border-radius: 11px !important;
    border-bottom-left-radius: 6px !important;
  }
  
  .typing-dots {
    display: flex;
    gap: 4px;
    align-items: center;
  }
  
  .typing-dots span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--dma-gray);
    animation: typing 1.4s infinite ease-in-out;
  }
  
  .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
  .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
  
  @keyframes typing {
    0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
    40% { transform: scale(1); opacity: 1; }
  }
  
  .message-content {
    display: flex;
    align-items: flex-start;
    gap: 8px;
  }
  
  .flag-icon {
    font-size: 16px;
    margin-top: 2px;
  }
  
  .language-indicator {
    font-size: 11px;
    color: var(--dma-gray);
    font-weight: 500;
    margin-top: 2px;
  }
  
  .message-text {
    flex: 1;
  }
  
  .message-time {
    font-size: 11px;
    color: var(--dma-gray);
    margin-top: 6px;
    text-align: right;
  }
  
  .msg.user .message-time {
    text-align: left;
  }
`;
document.head.appendChild(style);


