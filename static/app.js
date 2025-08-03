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
            "2. **Add your context** in the Knowledge Base on the left.\n" +
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
            this.updateContextStats();
            this.updateUI();
        });
        this.elements.chatInput.addEventListener('input', () => this.autoResizeTextarea(this.elements.chatInput));

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
     * Sets up the initial state of collapsible sections based on screen size.
     */
    setupResponsiveUI() {
        const isMobile = window.innerWidth < 1024;

        // On mobile, collapse the knowledge base by default to show the chat first.
        // On desktop, ensure everything is expanded.
        this.state.kbSectionCollapsed = isMobile;
        this.state.assistantSectionCollapsed = false; // Always show assistant on load

        // Hide API section by default if a valid key is already loaded
        if (this.state.apiKeyValidated) {
            this.state.apiSectionCollapsed = true;
        }

        this.updateSectionVisibility('api');
        this.updateSectionVisibility('kb');
        this.updateSectionVisibility('assistant');
    }

    /**
     * Toggles a specific collapsible section.
     * @param {'api' | 'kb' | 'assistant'} sectionName - The name of the section to toggle.
     */
    toggleSection(sectionName) {
        const stateKey = `${sectionName}SectionCollapsed`;
        this.state[stateKey] = !this.state[stateKey];
        this.updateSectionVisibility(sectionName);
    }

    /**
     * Updates the visibility of a collapsible section based on its state.
     * @param {'api' | 'kb' | 'assistant'} sectionName - The name of the section to update.
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
                // Don't auto-test on load, just update the UI
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

        // Reset validation state when input changes
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
            // Create an AbortController for timeout handling
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

            // CHANGE: Updated the endpoint to match the backend: /api/v1/test_api_key
            const response = await fetch('/api/v1/test_api_key', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ api_key: apiKey }),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server error (${response.status}): ${errorText}`);
            }

            const result = await response.json();

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
                if (!silent) {
                    this.addMessageToChat(`❌ **API Key Invalid**: ${result.message || 'Unknown error'}`, 'system');
                }
            }
        } catch (error) {
            console.error('API key test error:', error);
            this.state.apiKeyValidated = false;
            this.state.userApiKey = '';

            let errorMessage = 'Error testing API key';
            if (error.name === 'AbortError') {
                errorMessage = 'Request timed out. Please check your connection and try again.';
            } else if (error.message) {
                errorMessage = error.message;
            }

            this.updateApiKeyStatus('error', errorMessage);
            if (!silent) {
                this.addMessageToChat(`❌ **Connection Error**: ${errorMessage}`, 'system');
            }
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

        switch (status) {
            case 'testing':
                iconEl.className = 'w-3 h-3 bg-blue-500 rounded-full animate-pulse flex-shrink-0';
                statusEl.textContent = 'Testing...';
                break;
            case 'success':
                iconEl.className = 'w-3 h-3 bg-green-500 rounded-full flex-shrink-0';
                statusEl.textContent = 'API Key Valid';
                break;
            case 'error':
                iconEl.className = 'w-3 h-3 bg-red-500 rounded-full flex-shrink-0';
                statusEl.textContent = 'API Key Invalid';
                break;
            case 'pending':
                iconEl.className = 'w-3 h-3 bg-yellow-500 rounded-full flex-shrink-0';
                statusEl.textContent = 'API Key Pending';
                break;
        }

        // Also update the detailed status message box
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
     * Main handler for the send button. Directs to the correct function based on the selected task.
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
     * Handles the logic for indexing the provided context.
     */
    async handleIndexContext() {
        if (!this.state.apiKeyValidated) {
            this.addMessageToChat("🔑 **API Key Required**: Please validate your API key before indexing.", 'system');
            return;
        }

        const context = this.elements.contextInput.value.trim();
        if (context.length < 20) {
            this.showStatus('Context is too short. Please provide at least 20 characters.', 'error');
            return;
        }

        this.state.isIndexing = true;
        this.updateUI();
        this.showStatus('Indexing context... This may take a moment.', 'loading');

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout for indexing

            const response = await fetch('/api/v1/index', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': this.state.userApiKey
                },
                body: JSON.stringify({ context }),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server error (${response.status}): ${errorText}`);
            }

            const result = await response.json();
            this.state.isIndexed = true;
            // CHANGE: Updated expected response field from documents_added to chunks_added
            this.showStatus(`Successfully indexed ${result.chunks_added || '1'} document chunks.`, 'success');
        } catch (error) {
            console.error('Indexing error:', error);
            let errorMessage = 'Error indexing context';
            if (error.name === 'AbortError') {
                errorMessage = 'Indexing timed out. Please try with smaller content or check your connection.';
            } else if (error.message) {
                errorMessage = error.message;
            }
            this.showStatus(errorMessage, 'error');
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
            const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout

            // CHANGE: Updated the endpoint to match the backend: /api/v1/chat
            const response = await fetch('/api/v1/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': this.state.userApiKey
                },
                body: JSON.stringify({ prompt }),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorBody = await response.json().catch(() => ({ detail: 'Unknown error occurred' }));
                throw new Error(errorBody.detail || 'An unknown error occurred.');
            }

            const result = await response.json();
            this.addMessageToChat(result.response, 'ai');
            this.showStatus('Ready for your next question.', 'success');
        } catch (error) {
            console.error('Generation error:', error);
            let errorMessage = error.message;
            if (error.name === 'AbortError') {
                errorMessage = 'Request timed out. Please try again.';
            }
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
        if (prompt) {
            userMessage += `\nPrompt: ${prompt}`;
        }
        this.addMessageToChat(userMessage, 'user');

        this.elements.chatInput.value = '';
        this.autoResizeTextarea(this.elements.chatInput);

        this.state.isGenerating = true;
        this.updateUI();
        this.showStatus(`AI is performing task: ${task_type}...`, 'loading');

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 180000); // 60 second timeout

            // CHANGE: Updated the JSON body to match the new `TaskRequest` schema
            // which only expects `task_name`. The backend handles the rest.
            const response = await fetch('/api/v1/task', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': this.state.userApiKey
                },
                body: JSON.stringify({ task_name: task_type }),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorBody = await response.json().catch(() => ({ detail: 'Unknown error occurred' }));
                throw new Error(errorBody.detail || 'An unknown error occurred.');
            }

            const result = await response.json();
            this.addMessageToChat(result.result, 'ai');
            this.showStatus('Task completed successfully.', 'success');
        } catch (error) {
            console.error('Task execution error:', error);
            let errorMessage = error.message;
            if (error.name === 'AbortError') {
                errorMessage = 'Task timed out. Please try again.';
            }
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
        this.updateContextStats();
        this.state.isIndexed = false;
        this.updateUI();
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
        }
    }

    /**
     * Updates all UI elements based on the current application state.
     */
    updateUI() {
        const hasContext = this.elements.contextInput.value.trim().length > 10;
        const isQandA = this.elements.taskSelect.value === 'q_and_a';
        const hasValidApiKey = this.state.apiKeyValidated;
        const isBusy = this.state.isIndexing || this.state.isGenerating || this.state.isTestingApiKey;

        // Update API key related buttons
        const apiKeyEntered = this.elements.apiKeyInput.value.trim().length > 0;
        this.elements.testApiKeyBtn.disabled = this.state.isTestingApiKey || !apiKeyEntered;
        this.elements.saveApiKeyBtn.disabled = isBusy || !this.state.apiKeyValidated;

        // Update button text based on state
        if (this.state.isTestingApiKey) {
            this.elements.testApiKeyBtn.textContent = 'Testing...';
        } else {
            this.elements.testApiKeyBtn.textContent = 'Test Key';
        }

        // Update context and chat related buttons
        this.elements.indexContextBtn.disabled = isBusy || !hasContext || !hasValidApiKey;

        if (isQandA) {
            this.elements.sendButton.disabled = isBusy || !this.state.isIndexed || !hasValidApiKey;
        } else {
            this.elements.sendButton.disabled = isBusy || !hasContext || !hasValidApiKey;
        }

        // Update visual states
        const buttonStates = [
            this.elements.testApiKeyBtn,
            this.elements.saveApiKeyBtn,
            this.elements.indexContextBtn,
            this.elements.sendButton
        ];

        buttonStates.forEach(button => {
            if (button && button.disabled) {
                button.style.opacity = '0.5';
                button.style.cursor = 'not-allowed';
            } else if (button) {
                button.style.opacity = '1';
                button.style.cursor = 'pointer';
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

        indicator.innerHTML = `<span class="${colorClass}">${message}</span>`;

        if (type !== 'loading') {
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