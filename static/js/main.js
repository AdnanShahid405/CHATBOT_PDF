// State Management
const state = {
    documentLoaded: false,
    currentFile: null,
    processing: false
};

// DOM Elements
const elements = {
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    browseBtn: document.getElementById('browseBtn'),
    fileInfo: document.getElementById('fileInfo'),
    fileName: document.getElementById('fileName'),
    fileStats: document.getElementById('fileStats'),
    removeFileBtn: document.getElementById('removeFileBtn'),
    uploadProgress: document.getElementById('uploadProgress'),
    progressFill: document.getElementById('progressFill'),
    progressText: document.getElementById('progressText'),
    welcomeScreen: document.getElementById('welcomeScreen'),
    chatArea: document.getElementById('chatArea'),
    messages: document.getElementById('messages'),
    messageInput: document.getElementById('messageInput'),
    sendBtn: document.getElementById('sendBtn'),
    statusIndicator: document.getElementById('statusIndicator'),
    chunksCount: document.getElementById('chunksCount'),
    charsCount: document.getElementById('charsCount')
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    checkAPIHealth();
});

function initializeEventListeners() {
    // File upload events
    elements.browseBtn.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', handleFileSelect);
    elements.removeFileBtn.addEventListener('click', clearFile);
    
    // Drag and drop
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('dragleave', handleDragLeave);
    elements.uploadArea.addEventListener('drop', handleDrop);
    elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
    
    // Chat events
    elements.sendBtn.addEventListener('click', sendMessage);
    elements.messageInput.addEventListener('keydown', handleKeyDown);
    elements.messageInput.addEventListener('input', autoResizeTextarea);
}

// API Health Check
async function checkAPIHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (data.status === 'running') {
            updateStatus('online', 'Connected');
            if (data.documents_loaded) {
                state.documentLoaded = true;
                enableChat();
            }
        }
    } catch (error) {
        updateStatus('offline', 'Offline');
        showToast('Failed to connect to server', 'error');
    }
}

function updateStatus(status, text) {
    const dot = elements.statusIndicator.querySelector('.status-dot');
    const statusText = elements.statusIndicator.querySelector('.status-text');
    
    dot.className = 'status-dot';
    if (status === 'online') dot.classList.add('online');
    statusText.textContent = text;
}

// File Upload Handlers
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) uploadFile(file);
}

function handleDragOver(e) {
    e.preventDefault();
    elements.uploadArea.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type === 'application/pdf') {
        uploadFile(file);
    } else {
        showToast('Please upload a PDF file', 'error');
    }
}

async function uploadFile(file) {
    if (state.processing) return;
    
    state.processing = true;
    state.currentFile = file;
    
    // Show file info
    elements.uploadArea.style.display = 'none';
    elements.fileInfo.style.display = 'flex';
    elements.fileName.textContent = file.name;
    elements.fileStats.textContent = `${(file.size / 1024).toFixed(2)} KB`;
    
    // Show progress
    elements.uploadProgress.style.display = 'block';
    elements.progressFill.style.width = '0%';
    elements.progressText.textContent = 'Uploading...';
    
    // Simulate upload progress
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += 10;
        if (progress <= 90) {
            elements.progressFill.style.width = progress + '%';
        }
    }, 200);
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        clearInterval(progressInterval);
        
        if (response.ok) {
            const data = await response.json();
            
            // Complete progress
            elements.progressFill.style.width = '100%';
            elements.progressText.textContent = 'Processing complete!';
            
            // Update stats
            elements.chunksCount.textContent = data.chunks_created;
            elements.charsCount.textContent = data.total_characters.toLocaleString();
            
            // Enable chat
            state.documentLoaded = true;
            enableChat();
            
            showToast('Document uploaded successfully!', 'success');
            
            setTimeout(() => {
                elements.uploadProgress.style.display = 'none';
            }, 2000);
        } else {
            const error = await response.json();
            throw new Error(error.error || 'Upload failed');
        }
    } catch (error) {
        clearInterval(progressInterval);
        elements.uploadProgress.style.display = 'none';
        showToast(error.message, 'error');
        clearFile();
    } finally {
        state.processing = false;
    }
}

function clearFile() {
    state.currentFile = null;
    state.documentLoaded = false;
    
    elements.uploadArea.style.display = 'block';
    elements.fileInfo.style.display = 'none';
    elements.uploadProgress.style.display = 'none';
    elements.fileInput.value = '';
    
    // Clear chat
    elements.messages.innerHTML = '';
    elements.welcomeScreen.style.display = 'flex';
    elements.chatArea.style.display = 'none';
    
    disableChat();
}

function enableChat() {
    elements.welcomeScreen.style.display = 'none';
    elements.chatArea.style.display = 'flex';
    elements.messageInput.disabled = false;
    elements.sendBtn.disabled = false;
    elements.messageInput.placeholder = 'Ask a question about your document...';
    elements.messageInput.focus();
}

function disableChat() {
    elements.messageInput.disabled = true;
    elements.sendBtn.disabled = true;
    elements.messageInput.placeholder = 'Upload a document first...';
}

// Chat Functionality
async function sendMessage() {
    const question = elements.messageInput.value.trim();
    if (!question || state.processing) return;
    
    // Add user message
    addMessage(question, 'user');
    elements.messageInput.value = '';
    autoResizeTextarea();
    
    // Show typing indicator
    const typingId = addTypingIndicator();
    
    state.processing = true;
    elements.sendBtn.disabled = true;
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question })
        });
        
        removeTypingIndicator(typingId);
        
        if (response.ok) {
            const data = await response.json();
            addMessage(data.answer, 'bot', data.sources);
        } else {
            const error = await response.json();
            throw new Error(error.error || 'Failed to get response');
        }
    } catch (error) {
        removeTypingIndicator(typingId);
        addMessage('Sorry, I encountered an error: ' + error.message, 'bot');
        showToast(error.message, 'error');
    } finally {
        state.processing = false;
        elements.sendBtn.disabled = false;
        elements.messageInput.focus();
    }
}

function addMessage(text, type, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${type}`;
    
    let sourcesHTML = '';
    if (sources && sources.length > 0) {
        sourcesHTML = `
            <div class="message-sources">
                <h4>📚 Sources:</h4>
                ${sources.map((source, i) => `
                    <div class="source-item">
                        <strong>Source ${i + 1}</strong> 
                        <span class="source-similarity">(${(source.similarity * 100).toFixed(1)}% match)</span>
                        <p>${source.text}</p>
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="message-text">${escapeHtml(text)}</div>
            ${sourcesHTML}
        </div>
    `;
    
    elements.messages.appendChild(messageDiv);
    elements.messages.scrollTop = elements.messages.scrollHeight;
}

function addTypingIndicator() {
    const id = 'typing-' + Date.now();
    const typingDiv = document.createElement('div');
    typingDiv.id = id;
    typingDiv.className = 'message message-bot';
    typingDiv.innerHTML = `
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    
    elements.messages.appendChild(typingDiv);
    elements.messages.scrollTop = elements.messages.scrollHeight;
    return id;
}

function removeTypingIndicator(id) {
    const element = document.getElementById(id);
    if (element) element.remove();
}

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function autoResizeTextarea() {
    elements.messageInput.style.height = 'auto';
    elements.messageInput.style.height = elements.messageInput.scrollHeight + 'px';
}

// Toast Notifications
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        info: 'fa-info-circle'
    };
    
    toast.innerHTML = `
        <i class="fas ${icons[type]}"></i>
        <span>${message}</span>
    `;
    
    document.getElementById('toastContainer').appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Utility Functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}