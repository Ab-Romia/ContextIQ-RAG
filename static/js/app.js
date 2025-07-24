class ContextIQApp {
    constructor() {
        this.elements = {};
        this.state = {};
        this.initialize();
    }

    initialize() {
        this.initializeElements();
        this.initializeState();
        this.initializeEventListeners();
        this.addWelcomeMessage();
        this.updateContextStats();
    }

    initializeElements() {
        this.elements = {
            contextInput: document.getElementById('context-input'),
            chatInput: document.getElementById('chat-input'),
            sendButton: document.getElementById('send-button'),
            chatContainer: document.getElementById('chat-container'),
            typingIndicator: document.getElementById('typing-indicator'),
            responseTime: document.getElementById('response-time'),
            clearContextButton: document.getElementById('clear-context'),
            charCount: document.getElementById('char-count'),
            wordCount: document.getElementById('word-count'),
        };
    }

    initializeState() {
        this.state = {
            isProcessing: false
        };
        this.elements.sendButton.disabled = true;
    }

    initializeEventListeners() {
        this.elements.sendButton.addEventListener('click', () => this.handleSendPrompt());
        this.elements.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleSendPrompt();
            }
        });
        this.elements.chatInput.addEventListener('input', (e) => {
            this.autoResizeTextarea(e.target);
            this.validateInputs();
        });
        this.elements.contextInput.addEventListener('input', () => {
            this.updateContextStats();
            this.validateInputs();
        });
        this.elements.clearContextButton.addEventListener('click', () => this.clearContext());
    }

    autoResizeTextarea(element) {
        element.style.height = 'auto';
        element.style.height = Math.min(element.scrollHeight, 120) + 'px';
    }

    updateContextStats() {
        const text = this.elements.contextInput.value;
        const charCount = text.length;
        const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
        this.elements.charCount.textContent = charCount.toLocaleString();
        this.elements.wordCount.textContent = wordCount.toLocaleString();
    }

    validateInputs() {
        const context = this.elements.contextInput.value.trim();
        const prompt = this.elements.chatInput.value.trim();
        const isValid = context.length > 10 && prompt.length > 2 && !this.state.isProcessing;
        this.elements.sendButton.disabled = !isValid;
    }

    async handleSendPrompt() {
        if (this.elements.sendButton.disabled) return;

        const context = this.elements.contextInput.value.trim();
        const prompt = this.elements.chatInput.value.trim();

        this.addMessageToChat(prompt, 'user');
        this.elements.chatInput.value = '';
        this.autoResizeTextarea(this.elements.chatInput);

        this.setProcessing(true);
        const startTime = Date.now();

        try {
            // First index the context
            await this.indexContext(context);

            // Then call the generate API
            const aiResponse = await this.callBackendAPI(context, prompt);
            this.addMessageToChat(aiResponse, 'ai');
        } catch (error) {
            this.addMessageToChat(error.message, 'system');
        } finally {
            this.setProcessing(false);
            const responseTime = Date.now() - startTime;
            this.showResponseTime(responseTime);
            this.validateInputs();
        }
    }

    async indexContext(context) {
        // Only index if context has meaningful content
        if (context.length < 20) return;

        const apiUrl = '/api/v1/index';
        const payload = { context: context };

        await fetch(apiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
    }
    setProcessing(isProcessing) {
        this.state.isProcessing = isProcessing;
        this.elements.typingIndicator.classList.toggle('hidden', !isProcessing);
        this.elements.sendButton.disabled = isProcessing;
    }

    async callBackendAPI(context, userPrompt) {
        const apiUrl = '/api/v1/generate';
        const payload = {
            context: context,
            prompt: userPrompt,
        };

        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorBody = await response.json();
            throw new Error(`Server Error (${response.status}): ${errorBody.detail || 'An unknown error occurred.'}`);
        }

        const result = await response.json();
        return result.response;
    }

    addMessageToChat(message, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message flex items-start space-x-3 animate-fade-in';
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        let icon, bubbleClass;
        if (sender === 'user') {
            icon = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`;
            bubbleClass = 'bg-gradient-to-br from-emerald-500 to-teal-600';
        } else if (sender === 'ai') {
            icon = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>`;
            bubbleClass = 'bg-gradient-to-br from-indigo-500 to-purple-600';
        } else { // system/error
            icon = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
            bubbleClass = 'bg-gradient-to-br from-red-500 to-orange-600';
        }

        const formattedMessage = this.escapeHtml(message).replace(/\n/g, '<br>');

        messageDiv.innerHTML = `
            <div class="w-8 h-8 ${bubbleClass} rounded-full flex items-center justify-center flex-shrink-0 text-white">${icon}</div>
            <div class="flex-1">
                <div class="bg-slate-800/50 rounded-xl p-4 border border-slate-600/30">
                    <p class="text-slate-200">${formattedMessage}</p>
                </div>
                <div class="text-xs text-slate-500 mt-2">${timestamp}</div>
            </div>`;

        this.elements.chatContainer.appendChild(messageDiv);
        this.elements.chatContainer.scrollTop = this.elements.chatContainer.scrollHeight;
    }

    addWelcomeMessage() {
         this.addMessageToChat(`Welcome to ContextIQ! Here's how to get started:\n1. Paste your documents or context in the left panel.\n2. Ask me questions about the content.\n3. Get intelligent, source-backed responses.`, 'ai');
    }

    showResponseTime(time) {
        const el = this.elements.responseTime;
        el.querySelector('span').textContent = `${time}ms`;
        el.classList.remove('hidden');
        setTimeout(() => el.classList.add('hidden'), 4000);
    }

    clearContext() {
        this.elements.contextInput.value = '';
        this.updateContextStats();
        this.validateInputs();
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

}

document.addEventListener('DOMContentLoaded', () => {
    new ContextIQApp();
});