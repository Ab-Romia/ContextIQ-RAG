class ContextAwareApp {
    constructor() {
        // Centralized DOM element references
        this.elements = {
            contextInput: document.getElementById('context-input'),
            chatInput: document.getElementById('chat-input'),
            sendButton: document.getElementById('send-button'),
            chatContainer: document.getElementById('chat-container'),
            statusIndicator: document.getElementById('status-indicator'),
            clearContextBtn: document.getElementById('clear-context-btn'),
            indexContextBtn: document.getElementById('index-context-btn'),
            taskSelect: document.getElementById('task-select'),
            charCount: document.getElementById('char-count'),
            wordCount: document.getElementById('word-count'),
        };
        // Application state
        this.state = {
            isIndexing: false,
            isGenerating: false,
            isIndexed: false,
        };
        this.initialize();
    }

    /**
     * Sets up the application, event listeners, and initial state.
     */
    initialize() {
        this.addEventListeners();
        this.addMessageToChat(
            "Welcome! Here's how to get started:\n1. Paste your context into the 'Knowledge Base' on the left.\n2. Click 'Index Context' for Q&A, or select a different action from the dropdown.\n3. Provide a prompt if needed and click the send button.",
            'ai'
        );
        this.updateUI();
    }

    /**
     * Binds all necessary event listeners to DOM elements.
     */
    addEventListeners() {
        this.elements.indexContextBtn.addEventListener('click', () => this.handleIndexContext());
        this.elements.clearContextBtn.addEventListener('click', () => this.handleClearContext());
        this.elements.sendButton.addEventListener('click', () => this.handleSubmit());
        this.elements.chatInput.addEventListener('keydown', e => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleSubmit();
            }
        });
        this.elements.contextInput.addEventListener('input', () => this.updateContextStats());
        this.elements.chatInput.addEventListener('input', () => this.autoResizeTextarea(this.elements.chatInput));
    }

    /**
     * Main handler for the send button. Directs to the correct function based on the selected task.
     */
    handleSubmit() {
        const selectedTask = this.elements.taskSelect.value;
        if (selectedTask === 'q_and_a') {
            this.handleSendPrompt();
        } else {
            this.handleExecuteTask();
        }
    }


    /**
     * Handles the logic for indexing the provided context.
     */
    async handleIndexContext() {
        const context = this.elements.contextInput.value.trim();
        if (context.length < 20) {
            this.showStatus('Context is too short. Please provide at least 20 characters.', 'error');
            return;
        }

        this.state.isIndexing = true;
        this.updateUI();
        this.showStatus('Indexing context... This may take a moment.', 'loading');

        try {
            const response = await fetch('/api/v1/index', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ context }),
            });

            if (!response.ok) throw new Error(`Server error: ${response.statusText}`);

            const result = await response.json();
            this.state.isIndexed = true;
            this.showStatus(`Successfully indexed ${result.documents_added} document chunks.`, 'success');
        } catch (error) {
            this.showStatus(`Error indexing context: ${error.message}`, 'error');
            this.state.isIndexed = false;
        } finally {
            this.state.isIndexing = false;
            this.updateUI();
        }
    }

    /**
     * Handles sending a user's prompt to the backend for a response.
     */
    async handleSendPrompt() {
        const prompt = this.elements.chatInput.value.trim();
        if (prompt.length < 2 || this.state.isGenerating || !this.state.isIndexed) {
            if (!this.state.isIndexed) {
                this.showStatus('Please index your context before asking questions.', 'error');
            }
            return;
        }

        this.addMessageToChat(prompt, 'user');
        this.elements.chatInput.value = '';
        this.autoResizeTextarea(this.elements.chatInput);

        this.state.isGenerating = true;
        this.updateUI();
        this.showStatus('AI is thinking...', 'loading');

        try {
            const response = await fetch('/api/v1/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt }),
            });

            if (!response.ok) {
                const errorBody = await response.json();
                throw new Error(errorBody.detail || 'An unknown error occurred.');
            }

            const result = await response.json();
            this.addMessageToChat(result.response, 'ai');
            this.showStatus('Ready for your next question.', 'success');
        } catch (error) {
            this.addMessageToChat(`An error occurred: ${error.message}`, 'system');
            this.showStatus(`Error: ${error.message}`, 'error');
        } finally {
            this.state.isGenerating = false;
            this.updateUI();
        }
    }

    /**
     * Handles executing a non-Q&A task like summarization or planning.
     */
    async handleExecuteTask() {
        const context = this.elements.contextInput.value.trim();
        const task_type = this.elements.taskSelect.value;
        const prompt = this.elements.chatInput.value.trim();

        if (context.length < 20) {
            this.showStatus('Please provide at least 20 characters of context for this task.', 'error');
            return;
        }

        let userMessage = `Task: ${task_type}`;
        if (prompt) {
            userMessage += `\nPrompt: ${prompt}`;
        }
        this.addMessageToChat(userMessage, 'user');

        this.elements.chatInput.value = '';
        this.autoResizeTextarea(this.elements.chatInput);

        this.state.isGenerating = true;
        this.updateUI();
        this.showStatus('AI is performing the task...', 'loading');

        try {
            const response = await fetch('/api/v1/task', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ context, task_type, prompt }),
            });

            if (!response.ok) {
                const errorBody = await response.json();
                throw new Error(errorBody.detail || 'An unknown error occurred.');
            }

            const result = await response.json();
            this.addMessageToChat(result.result, 'ai');
            this.showStatus('Task completed successfully.', 'success');
        } catch (error) {
            this.addMessageToChat(`An error occurred: ${error.message}`, 'system');
            this.showStatus(`Error: ${error.message}`, 'error');
        } finally {
            this.state.isGenerating = false;
            this.updateUI();
        }
    }


    /**
     * Clears the context input and the indexed data on the backend.
     */
    async handleClearContext() {
        this.elements.contextInput.value = '';
        this.updateContextStats();
        this.state.isIndexed = false;
        this.updateUI();
        this.showStatus('Clearing knowledge base...', 'loading');

        try {
            await fetch('/api/v1/clear_index', { method: 'POST' });
            this.showStatus('Knowledge base cleared. Ready for new context.', 'success');
        } catch (error) {
            this.showStatus(`Error clearing index: ${error.message}`, 'error');
        }
    }

    /**
     * Updates all UI elements based on the current application state.
     */
    updateUI() {
        const hasContext = this.elements.contextInput.value.trim().length > 10;
        const isQandA = this.elements.taskSelect.value === 'q_and_a';

        this.elements.indexContextBtn.disabled = this.state.isIndexing || !hasContext;

        if (isQandA) {
            this.elements.sendButton.disabled = this.state.isGenerating || !this.state.isIndexed;
        } else {
            this.elements.sendButton.disabled = this.state.isGenerating || !hasContext;
        }
    }


    /**
     * Displays a status message to the user.
     * @param {string} message - The message to display.
     * @param {'loading'|'success'|'error'} type - The type of message.
     */
    showStatus(message, type) {
        const indicator = this.elements.statusIndicator;
        indicator.classList.remove('hidden');
        let colorClass = 'text-slate-400';
        if (type === 'success') colorClass = 'text-green-400';
        if (type === 'error') colorClass = 'text-red-400';

        indicator.innerHTML = `<span class="${colorClass}">${message}</span>`;

        // Hide the message after a delay unless it's a loading indicator
        if (type !== 'loading') {
            setTimeout(() => indicator.classList.add('hidden'), 5000);
        }
    }

    /**
     * Adds a new message to the chat display.
     * @param {string} message - The message content.
     * @param {'user'|'ai'|'system'} sender - The sender of the message.
     */
addMessageToChat(message, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message flex items-start space-x-3 animate-fade-in';
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const icons = {
        user: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`,
        ai: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>`,
        system: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`
    };
    const bubbleClasses = {
        user: 'bg-gradient-to-br from-emerald-500 to-teal-600',
        ai: 'bg-gradient-to-br from-indigo-500 to-purple-600',
        system: 'bg-gradient-to-br from-red-500 to-orange-600'
    };

    // Use marked to parse markdown, but sanitize first
    const formattedMessage = sender === 'user' ?
        this.escapeHtml(message).replace(/\n/g, '<br>') :
        marked.parse(message);

    messageDiv.innerHTML = `
        <div class="w-8 h-8 ${bubbleClasses[sender]} rounded-full flex items-center justify-center flex-shrink-0 text-white">${icons[sender]}</div>
        <div class="flex-1">
            <div class="bg-slate-800/50 rounded-xl p-4 border border-slate-600/30">
                <div class="text-slate-200 leading-relaxed markdown-content">${formattedMessage}</div>
            </div>
            <div class="text-xs text-slate-500 mt-2">${timestamp}</div>
        </div>`;

    this.elements.chatContainer.appendChild(messageDiv);
    this.elements.chatContainer.scrollTop = this.elements.chatContainer.scrollHeight;
}

    /**
     * Updates character and word counts for the context input.
     */
    updateContextStats() {
        const text = this.elements.contextInput.value;
        this.elements.charCount.textContent = text.length.toLocaleString();
        this.elements.wordCount.textContent = (text.trim().split(/\s+/).filter(Boolean).length).toLocaleString();
        this.updateUI();
    }

    autoResizeTextarea(element) {
        element.style.height = 'auto';
        element.style.height = `${Math.min(element.scrollHeight, 120)}px`;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new ContextAwareApp();
});