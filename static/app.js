class ContextAwareApp {
    constructor() {
        // Centralized DOM element references
        this.elements = {
            // Main layout panels
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

            // API Key elements
            apiKeyInput: document.getElementById('api-key-input'),
            testApiKeyBtn: document.getElementById('test-api-key'),
            saveApiKeyBtn: document.getElementById('save-api-key'),
            apiKeyStatus: document.getElementById('api-key-status'),
            apiStatusIcon: document.getElementById('api-status-icon'),
            apiStatusText: document.getElementById('api-status-text'),
            toggleApiSection: document.getElementById('toggle-api-section'),
            apiKeyContent: document.getElementById('api-key-content'),
            toggleIcon: document.getElementById('toggle-icon'),

            // File Input elements
            fileInput: document.getElementById('file-input'),
            fileName: document.getElementById('file-name'),

            // Responsive collapsible section elements
            kbHeader: document.getElementById('kb-header'),
            kbContent: document.getElementById('kb-content'),
            kbToggleIcon: document.getElementById('kb-toggle-icon'),
            assistantHeader: document.getElementById('assistant-header'),
            assistantContent: document.getElementById('assistant-content'),
            assistantToggleIcon: document.getElementById('assistant-toggle-icon'),
        };

        // Application state
        this.state = {
            isIndexing: false,
            isGenerating: false,
            isIndexed: false,
            apiKeyValidated: false,
            isTestingApiKey: false,
            userApiKey: '',
            // Collapse states for mobile view
            apiSectionCollapsed: false,
            kbSectionCollapsed: false,
            assistantSectionCollapsed: false,
        };

        this.initialize();
    }

    /**
     * Sets up the application, event listeners, and initial state.
     */
    async initialize() {
        this.addEventListeners();
        this.loadStoredApiKey();
        this.setupResponsiveUI();

        // Show welcome message
        this.addMessageToChat(
            "👋 **Welcome to ContextIQ!**\n\n" +
            "To get started:\n" +
            "1. **Enter your OpenRouter API key** in the configuration section above.\n" +
            "2. **Add your context** by uploading a file or pasting text in the Knowledge Base.\n" +
            "3. **Index the context** and start asking questions!\n\n" +
            "🆓 You can get a free API key from [openrouter.ai](https://openrouter.ai) - no credit card required!",
            'system'
        );

        // Initial UI update
        this.updateUI();
        this.updateContextStats();
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
        this.elements.contextInput.addEventListener('input', () => {
            // If user types in textarea, clear the file input as text takes precedence
            if (this.elements.fileInput.value) {
                this.elements.fileInput.value = '';
                this.elements.fileName.textContent = 'Choose a file...';
            }
            this.updateContextStats();
            this.updateUI();
        });
        this.elements.chatInput.addEventListener('input', () => this.autoResizeTextarea(this.elements.chatInput));

        // File input listener
        this.elements.fileInput.addEventListener('change', () => this.handleFileSelection());


        // API Key listeners
        this.elements.testApiKeyBtn.addEventListener('click', (e) => {
            e.preventDefault();
            this.testApiKey();
        });
        this.elements.saveApiKeyBtn.addEventListener('click', (e) => {
            e.preventDefault();
            this.saveApiKey();
        });
        this.elements.apiKeyInput.addEventListener('input', () => {
            this.onApiKeyInputChange();
            this.updateUI();
        });
        this.elements.apiKeyInput.addEventListener('keydown', e => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.testApiKey();
            }
        });

        // Toggle listeners for collapsible sections
        this.elements.toggleApiSection.addEventListener('click', () => this.toggleSection('api'));
        this.elements.kbHeader.addEventListener('click', () => this.toggleSection('kb'));
        this.elements.assistantHeader.addEventListener('click', () => this.toggleSection('assistant'));

        // Listen for window resize to adjust UI
        window.addEventListener('resize', () => this.setupResponsiveUI());
    }

    /**
     * Handles file selection, updates UI, and clears textarea.
     */
    handleFileSelection() {
        const file = this.elements.fileInput.files[0];
        if (file) {
            this.elements.fileName.textContent = file.name;
            // Clear textarea and its stats when a file is selected to indicate file is the source
            this.elements.contextInput.value = '';
            this.updateContextStats();
            this.updateUI();
        } else {
            this.elements.fileName.textContent = 'Choose a file...';
        }
    }


    /**
     * Sets up the initial state of collapsible sections based on screen size.
     */
    setupResponsiveUI() {
        const isMobile = window.innerWidth < 1024;

        this.state.kbSectionCollapsed = isMobile;
        this.state.assistantSectionCollapsed = false;

        if (this.state.apiKeyValidated) {
            this.state.apiSectionCollapsed = true;
        }

        this.updateSectionVisibility('api');
        this.updateSectionVisibility('kb');
        this.updateSectionVisibility('assistant');
    }

    /**
     * Toggles a specific collapsible section.
     */
    toggleSection(sectionName) {
        const stateKey = `${sectionName}SectionCollapsed`;
        this.state[stateKey] = !this.state[stateKey];
        this.updateSectionVisibility(sectionName);
    }

    /**
     * Updates the visibility of a collapsible section.
     */
    updateSectionVisibility(sectionName) {
        const contentEl = this.elements[`${sectionName}Content`];
        const toggleIconEl = this.elements[`${sectionName}ToggleIcon`];
        const isCollapsed = this.state[`${sectionName}SectionCollapsed`];

        if (contentEl) {
            contentEl.style.display = isCollapsed ? 'none' : 'block';
             if(sectionName !== 'api' && contentEl.classList.contains('lg:flex')){
                 contentEl.style.display = isCollapsed ? 'none' : 'flex';
             }
        }
        if (toggleIconEl) {
            toggleIconEl.style.transform = isCollapsed ? 'rotate(-90deg)' : 'rotate(0deg)';
        }
    }

    /**
     * Load stored API key from localStorage if available
     */
    loadStoredApiKey() {
        try {
            const storedKey = localStorage.getItem('openrouter_api_key');
            if (storedKey) {
                this.elements.apiKeyInput.value = storedKey;
                this.state.userApiKey = storedKey;
                this.onApiKeyInputChange();
            }
        } catch (error) {
            console.warn('Could not load stored API key:', error);
        }
    }

    /**
     * Handle API key input changes
     */
    onApiKeyInputChange() {
        const apiKey = this.elements.apiKeyInput.value.trim();

        this.state.apiKeyValidated = false;
        this.state.userApiKey = '';

        if (!apiKey) {
            this.updateApiKeyStatus('pending', 'Enter API key and click Test');
        } else if (!apiKey.startsWith('sk-or-')) {
            this.updateApiKeyStatus('error', 'Key should start with "sk-or-"');
        } else if (apiKey.length < 40) {
            this.updateApiKeyStatus('error', 'API key appears too short');
        } else {
            this.updateApiKeyStatus('pending', 'Click "Test Key" to validate');
        }
    }

    /**
     * Test the API key validity
     */
    async testApiKey(silent = false) {
        const apiKey = this.elements.apiKeyInput.value.trim();
        if (!apiKey) {
            if (!silent) this.updateApiKeyStatus('error', 'Please enter an API key');
            return;
        }

        this.state.isTestingApiKey = true;
        this.updateUI();
        if (!silent) {
            this.updateApiKeyStatus('testing', 'Testing API key...');
        }

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 120000);

            const response = await fetch('/api/v1/test-api-key', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ api_key: apiKey }),
                signal: controller.signal
            });

            clearTimeout(timeoutId);
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || `Server error (${response.status})`);

            if (result.valid) {
                this.state.apiKeyValidated = true;
                this.state.userApiKey = apiKey;
                this.updateApiKeyStatus('success', result.message || 'API key is valid');
                if (!silent) {
                    this.addMessageToChat("✅ **API Key Validated!** You can now use the assistant.", 'system');
                    this.state.apiSectionCollapsed = true;
                    this.updateSectionVisibility('api');
                }
            } else {
                this.state.apiKeyValidated = false;
                this.state.userApiKey = '';
                this.updateApiKeyStatus('error', result.message || 'API key is invalid');
                if (!silent) this.addMessageToChat(`❌ **API Key Invalid**: ${result.message || 'Unknown error'}`, 'system');
            }
        } catch (error) {
            console.error('API key test error:', error);
            this.state.apiKeyValidated = false;
            this.state.userApiKey = '';

            let errorMessage = (error.name === 'AbortError') ? 'Request timed out.' : error.message;
            this.updateApiKeyStatus('error', errorMessage);
            if (!silent) this.addMessageToChat(`❌ **Connection Error**: ${errorMessage}`, 'system');
        } finally {
            this.state.isTestingApiKey = false;
            this.updateUI();
        }
    }

    /**
     * Save the API key to localStorage
     */
    saveApiKey() {
        const apiKey = this.elements.apiKeyInput.value.trim();
        if (!this.state.apiKeyValidated) {
            this.addMessageToChat("⚠️ **Please test the API key first** before saving.", 'system');
            return;
        }
        try {
            localStorage.setItem('openrouter_api_key', apiKey);
            this.updateApiKeyStatus('success', 'API key saved locally!');
            this.addMessageToChat("💾 **API Key Saved!** It will be remembered for future sessions.", 'system');
        } catch (error) {
            console.error('Save error:', error);
            this.addMessageToChat("❌ **Save Failed**: Could not save API key to local storage.", 'system');
        }
    }

    /**
     * Update API key status display
     */
    updateApiKeyStatus(status, message) {
        const statusEl = this.elements.apiStatusText;
        const iconEl = this.elements.apiStatusIcon;

        const statusConfig = {
            testing: { icon: 'bg-blue-500 animate-pulse', text: 'Testing...' },
            success: { icon: 'bg-green-500', text: 'API Key Valid' },
            error:   { icon: 'bg-red-500', text: 'API Key Invalid' },
            pending: { icon: 'bg-yellow-500', text: 'API Key Pending' },
        };

        iconEl.className = `w-3 h-3 ${statusConfig[status].icon} rounded-full flex-shrink-0`;
        statusEl.textContent = statusConfig[status].text;

        const detailedStatusEl = this.elements.apiKeyStatus;
        if (detailedStatusEl) {
            detailedStatusEl.textContent = message;
            detailedStatusEl.classList.remove('hidden');
            detailedStatusEl.className = 'p-3 rounded-lg text-sm ';

            const colors = {
                testing: 'bg-blue-500/20 text-blue-300 border border-blue-500/30',
                success: 'bg-green-500/20 text-green-300 border border-green-500/30',
                error: 'bg-red-500/20 text-red-300 border border-red-500/30',
                pending: 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30',
            };
            detailedStatusEl.classList.add(...colors[status].split(' '));
        }
    }

    /**
     * Main handler for the send button.
     */
    handleSubmit() {
        if (!this.state.apiKeyValidated) {
            this.addMessageToChat("🔑 **API Key Required**: Please enter and test your API key first.", 'system');
            this.state.apiSectionCollapsed = false;
            this.updateSectionVisibility('api');
            this.elements.apiKeyInput.focus();
            return;
        }

        const selectedTask = this.elements.taskSelect.value;
        if (selectedTask === 'q_and_a') {
            this.handleSendPrompt();
        } else {
            this.handleExecuteTask();
        }
    }

    /**
     * ✨ REFACTORED: Unified logic for indexing from file or text.
     */
    async handleIndexContext() {
        if (this.state.isIndexing) return; // Prevent double-clicks

        if (!this.state.apiKeyValidated) {
            this.addMessageToChat("🔑 **API Key Required**: Please validate your API key before indexing.", 'system');
            return;
        }

        const file = this.elements.fileInput.files[0];
        const textContext = this.elements.contextInput.value.trim();

        if (!file && textContext.length < 20) {
            this.showStatus('Context is too short. Please provide at least 20 characters or upload a file.', 'error');
            return;
        }

        this.state.isIndexing = true;
        this.updateUI();
        this.showStatus(file ? `Uploading and indexing ${file.name}...` : 'Indexing text...', 'loading');

        try {
            let response;
            if (file) {
                // Handle file upload
                const formData = new FormData();
                formData.append('file', file);
                this.showStatus('Sending file to backend...', 'loading');
                response = await fetch('/api/v1/index-file', {
                    method: 'POST',
                    headers: { 'X-API-Key': this.state.userApiKey },
                    body: formData,
                });
            } else {
                // Handle text input
                this.showStatus('Sending text to backend...', 'loading');
                response = await fetch('/api/v1/index', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-API-Key': this.state.userApiKey
                    },
                    body: JSON.stringify({ context: textContext }),
                });
            }

            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.detail || 'An unknown error occurred during indexing.');
            }

            this.state.isIndexed = true;
            this.showStatus(result.message || 'Successfully indexed context.', 'success');

            // NEW: Populate textarea with extracted text if available
            if (result.extracted_text) {
                this.elements.contextInput.value = result.extracted_text;
                this.updateContextStats();
            }

        } catch (error) {
            console.error('Indexing error:', error);
            this.showStatus(`Error: ${error.message}`, 'error');
            this.state.isIndexed = false;
        } finally {
            this.state.isIndexing = false;
            this.updateUI();
            // Clear the file input after processing to prevent accidental re-uploads
            this.elements.fileInput.value = '';
            this.elements.fileName.textContent = 'Choose a file...';
        }
    }

    /**
     * Handles sending a user's prompt to the backend for a response.
     */
    async handleSendPrompt() {
        const prompt = this.elements.chatInput.value.trim();
        if (prompt.length < 2 || this.state.isGenerating) return;

        if (!this.state.isIndexed) {
            this.showStatus('Please index your context before asking questions.', 'error');
            return;
        }

        this.addMessageToChat(prompt, 'user');
        this.elements.chatInput.value = '';
        this.autoResizeTextarea(this.elements.chatInput);

        this.state.isGenerating = true;
        this.updateUI();
        this.showStatus('AI is thinking...', 'loading');

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 60000);

            const response = await fetch('/api/v1/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': this.state.userApiKey
                },
                body: JSON.stringify({ prompt }),
                signal: controller.signal
            });

            clearTimeout(timeoutId);
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || 'An unknown error occurred.');

            this.addMessageToChat(result.response, 'ai');
            this.showStatus('Ready for your next question.', 'success');
        } catch (error) {
            console.error('Generation error:', error);
            const errorMessage = (error.name === 'AbortError') ? 'Request timed out. Please try again.' : error.message;
            this.addMessageToChat(`An error occurred: ${errorMessage}`, 'system');
            this.showStatus(`Error: ${errorMessage}`, 'error');
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

        let userMessage = `Task: ${task_type.charAt(0).toUpperCase() + task_type.slice(1)}`;
        if (prompt) userMessage += `\nPrompt: ${prompt}`;
        this.addMessageToChat(userMessage, 'user');

        this.elements.chatInput.value = '';
        this.autoResizeTextarea(this.elements.chatInput);

        this.state.isGenerating = true;
        this.updateUI();
        this.showStatus(`AI is performing task: ${task_type}...`, 'loading');

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 120000);

            const response = await fetch('/api/v1/task', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': this.state.userApiKey
                },
                body: JSON.stringify({ context, task_type, prompt }),
                signal: controller.signal
            });

            clearTimeout(timeoutId);
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || 'An unknown error occurred.');

            this.addMessageToChat(result.result, 'ai');
            this.showStatus('Task completed successfully.', 'success');
        } catch (error) {
            console.error('Task execution error:', error);
            const errorMessage = (error.name === 'AbortError') ? 'Task timed out. Please try again.' : error.message;
            this.addMessageToChat(`An error occurred: ${errorMessage}`, 'system');
            this.showStatus(`Error: ${errorMessage}`, 'error');
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
        this.elements.fileInput.value = ''; // Also clear the file input
        this.elements.fileName.textContent = 'Choose a file...';
        this.updateContextStats();
        this.state.isIndexed = false;

        this.showStatus('Clearing knowledge base...', 'loading');

        try {
            await fetch('/api/v1/clear_index', {
                method: 'POST',
                headers: { 'X-API-Key': this.state.userApiKey }
            });
            this.showStatus('Knowledge base cleared. Ready for new context.', 'success');
        } catch (error) {
            console.error('Clear index error:', error);
            this.showStatus(`Error clearing index: ${error.message}`, 'error');
        } finally {
            this.updateUI();
        }
    }

    /**
     * Updates all UI elements based on the current application state.
     */
    updateUI() {
        const hasTextContext = this.elements.contextInput.value.trim().length > 10;
        const hasFileContext = this.elements.fileInput.files.length > 0;
        const hasContext = hasTextContext || hasFileContext;

        const isQandA = this.elements.taskSelect.value === 'q_and_a';
        const hasValidApiKey = this.state.apiKeyValidated;
        const isBusy = this.state.isIndexing || this.state.isGenerating || this.state.isTestingApiKey;

        // Update API key related buttons
        const apiKeyEntered = this.elements.apiKeyInput.value.trim().length > 0;
        this.elements.testApiKeyBtn.disabled = this.state.isTestingApiKey || !apiKeyEntered;
        this.elements.saveApiKeyBtn.disabled = isBusy || !this.state.apiKeyValidated;
        this.elements.testApiKeyBtn.textContent = this.state.isTestingApiKey ? 'Testing...' : 'Test Key';

        // Update context and chat related buttons
        this.elements.indexContextBtn.disabled = isBusy || !hasContext || !hasValidApiKey;

        if (isQandA) {
            this.elements.sendButton.disabled = isBusy || !this.state.isIndexed || !hasValidApiKey;
        } else {
            // For other tasks, context comes from the text area, not the index
            this.elements.sendButton.disabled = isBusy || !hasTextContext || !hasValidApiKey;
        }

        // Update visual states for all buttons
        [this.elements.testApiKeyBtn, this.elements.saveApiKeyBtn, this.elements.indexContextBtn, this.elements.sendButton].forEach(button => {
            if (button) {
                button.style.opacity = button.disabled ? '0.5' : '1';
                button.style.cursor = button.disabled ? 'not-allowed' : 'pointer';
            }
        });
    }

    /**
     * Displays a status message to the user.
     */
    showStatus(message, type) {
        const indicator = this.elements.statusIndicator;
        indicator.classList.remove('hidden');
        let colorClass = 'text-slate-400';
        if (type === 'success') colorClass = 'text-green-400';
        if (type === 'error') colorClass = 'text-red-400';
        if (type === 'loading') colorClass = 'text-blue-400 animate-pulse';

        indicator.innerHTML = `<span class="${colorClass}">${message}</span>`;

        if (type !== 'loading') {
            indicator.querySelector('span').classList.remove('animate-pulse');
            setTimeout(() => indicator.classList.add('hidden'), 5000);
        }
    }

    /**
     * Adds a new message to the chat display.
     */
    addMessageToChat(message, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message flex items-start space-x-3 animate-fade-in';
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        const icons = {
            user: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`,
            ai: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>`,
            system: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>`
        };
        const bubbleClasses = {
            user: 'bg-gradient-to-br from-emerald-500 to-teal-600',
            ai: 'bg-gradient-to-br from-indigo-500 to-purple-600',
            system: 'bg-gradient-to-br from-blue-500 to-cyan-600'
        };

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

// Initialize the app when DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    new ContextAwareApp();
});