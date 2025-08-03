# ContextIQ RAG - Intelligent Context-Aware Assistant

ContextIQ is a sophisticated **Retrieval-Augmented Generation (RAG)** application that allows users to seamlessly upload and interact with their own text content. It provides a powerful AI assistant capable of answering questions, summarizing information, and performing specialized tasks based on the provided documents.

## 🌟 Features

-   **📄 Document Indexing**: Easily index text content to create a personalized knowledge base for the AI assistant.
-   **🤖 Context-Aware Q&A**: Get intelligent, contextually relevant answers to questions directly from your indexed documents.
-   **📝 Specialized Task Execution**: The assistant can perform a variety of tasks such as summarization, generating action plans, and creative writing.
-   **🔍 Vector Search**: Powered by **ChromaDB**, the backend uses a custom TF-IDF-based embedding function for efficient and relevant document retrieval.
-   **⚡ Real-time Processing**: Experience fast vector search and AI response generation.
-   **🎨 Modern UI**: A beautiful, responsive user interface is provided, featuring a dark theme and smooth animations.
-   **💾 Intelligent Caching**: The system includes a caching mechanism to store recent responses, improving performance and reducing redundant API calls.
-   **🔑 API Key Management**: The frontend allows users to enter, test, and save their own OpenRouter API key directly, providing flexibility and control.

## 🏗️ Architecture

ContextIQ is built with a clear separation of concerns, featuring a Python backend powered by **FastAPI** and a modern JavaScript frontend.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   OpenRouter    │
│   (HTML/JS)     │◄──►│   Backend       │◄──►│   AI Models     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   ChromaDB      │
                       │ Vector Database │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

-   Python 3.8 or higher
-   An **OpenRouter API key** (You can get a free one [here](https://openrouter.ai/))

### Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/Ab-Romia/ContextIQ-RAG.git](https://github.com/Ab-Romia/ContextIQ-RAG.git)
    cd ContextIQ-RAG
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3. **API Configuration**

   This application doesn't require setting environment variables in a `.env` file. Instead:
   
   - When you run the application, you'll be prompted to enter your OpenRouter API key in the UI
   - Get a free API key from [openrouter.ai](https://openrouter.ai)
   - The API key is stored locally in your browser and sent with your requests
   - You can test the API key's validity directly in the interface

### Usage

1.  **Run the application**
    ```bash
    uvicorn main:app --reload
    ```
    This will start the FastAPI server.

2.  **Access the web interface**
    Open your browser and navigate to `http://127.0.0.1:8000`.

3.  **Interact with the AI**
    * In the web interface, enter your OpenRouter API key.
    * Paste your text content into the "Knowledge Base" section and click "Index Context".
    * Ask questions or select a task for the AI to perform based on your indexed text.


---

**Built by Ab-Romia** | **Refactored for Clarity & Efficiency**
