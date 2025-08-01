# ContextIQ RAG - Intelligent Context-Aware Assistant

A sophisticated Retrieval-Augmented Generation (RAG) application that allows users to upload context documents and interact with them through an AI assistant. Built with FastAPI backend and modern JavaScript frontend, featuring ChromaDB for vector storage and OpenRouter for AI completions.

## 🌟 Features

- **📄 Document Indexing**: Upload and index any text content for AI-powered interactions
- **🤖 Context-Aware Q&A**: Ask questions about your uploaded documents with intelligent responses
- **📝 Task Execution**: Perform specialized tasks like summarization, planning, and creative writing
- **⚡ Real-time Processing**: Fast vector search and AI response generation
- **🎨 Modern UI**: Beautiful, responsive interface with dark theme and animations
- **💾 Smart Caching**: Intelligent response caching to improve performance
- **🔍 Vector Search**: Powered by ChromaDB with SentenceTransformer embeddings

## 🏗️ Architecture

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

- Python 3.8 or higher
- OpenRouter API key ([Get one here](https://openrouter.ai/))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ab-Romia/ContextIQ-RAG.git
   cd ContextIQ-RAG
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   MODEL_NAME=deepseek/deepseek-r1-0528:free
   OPENROUTER_URL=https://openrouter.ai/api/v1
   ```

4. **Run the application**
   ```bash
   python -m app.main
   ```

5. **Open your browser**
   
   Navigate to `http://127.0.0.1:8000` to start using ContextIQ!

## 📖 Usage Guide

### 1. Index Your Content
- Paste your documents, meeting notes, or any text content into the "Knowledge Base" panel
- Click "Index Context" to process and store the content in the vector database
- Wait for the "Successfully indexed" confirmation

### 2. Choose Your Action
Select from the dropdown menu:
- **Question & Answer**: Ask specific questions about your indexed content
- **Summarize**: Generate concise summaries of your documents
- **Generate Action Plan**: Create detailed action plans based on your content
- **Creative Writing**: Use your content as inspiration for creative pieces

### 3. Interact with the AI
- For Q&A: Type your question and click send
- For tasks: Optionally provide additional prompts to guide the AI
- View responses in the chat interface with markdown formatting

## 🛠️ API Reference

### Core Endpoints

#### Index Document
```http
POST /api/v1/index
Content-Type: application/json

{
  "context": "Your document content here..."
}
```

#### Generate Response
```http
POST /api/v1/generate
Content-Type: application/json

{
  "prompt": "Your question here"
}
```

#### Execute Task
```http
POST /api/v1/task
Content-Type: application/json

{
  "context": "Document content",
  "task_type": "summarize|plan|creative",
  "prompt": "Optional guidance prompt"
}
```

#### Clear Index
```http
POST /api/v1/clear_index
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | Required |
| `MODEL_NAME` | AI model to use | `deepseek/deepseek-r1-0528:free` |
| `OPENROUTER_URL` | OpenRouter API base URL | `https://openrouter.ai/api/v1` |

### Supported Models

The application works with any OpenRouter-compatible model. Popular choices:
- `deepseek/deepseek-r1-0528:free` (Free tier)
- `anthropic/claude-3-haiku`
- `openai/gpt-3.5-turbo`
- `meta-llama/llama-3.1-8b-instruct:free`

## 📁 Project Structure

```
ContextIQ-RAG/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── config.py         # Configuration management
│   ├── schemas.py        # Pydantic models
│   ├── services.py       # Business logic
│   └── rag_setup.py      # RAG initialization
├── static/
│   └── app.js           # Frontend JavaScript
├── templates/
│   └── index.html       # Main UI template
├── .env                 # Environment variables
└── requirements.txt     # Python dependencies
```

## ⚙️ Technical Details

### Vector Database
- **ChromaDB**: Persistent vector storage with automatic embedding generation
- **Embeddings**: SentenceTransformer for high-quality text embeddings
- **Chunking**: Intelligent text chunking for optimal retrieval

### AI Integration
- **OpenRouter**: Access to multiple AI models through a single API
- **Async Processing**: Non-blocking AI request handling
- **Error Handling**: Robust retry logic and graceful error management

### Frontend Features
- **Responsive Design**: Mobile-friendly interface with Tailwind CSS
- **Real-time Updates**: Live status indicators and progress feedback
- **Markdown Support**: Rich text rendering with marked.js
- **Auto-resize**: Dynamic textarea sizing for better UX

## 🔒 Security & Privacy

- **Local Processing**: Document embeddings are generated locally
- **No Data Persistence**: Conversations are not stored permanently
- **API Key Protection**: Environment-based configuration
- **Input Validation**: Comprehensive request validation with Pydantic

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


---

**Built by Ab-Romia** | **Refactored for Clarity & Efficiency**
