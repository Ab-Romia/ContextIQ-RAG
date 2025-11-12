# 🧠 ContextIQ - Intelligent Context-Aware AI Assistant

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/spaces/Ab-Romia/Context-Aware-AI)

**A sophisticated RAG (Retrieval-Augmented Generation) application powered by multiple AI providers**

[Live Demo](https://huggingface.co/spaces/Ab-Romia/Context-Aware-AI) · [Report Bug](https://github.com/Ab-Romia/ContextIQ-RAG/issues) · [Request Feature](https://github.com/Ab-Romia/ContextIQ-RAG/issues)

</div>

---

## 🌟 What is ContextIQ?

ContextIQ is an advanced **Retrieval-Augmented Generation (RAG)** application that transforms how you interact with your documents. Upload any document, ask questions, get summaries, or generate insights - all powered by state-of-the-art AI models from **OpenAI** and **OpenRouter**.

### ✨ Key Highlights

- 🎯 **Dual AI Provider Support**: Choose between OpenAI (GPT-4o, GPT-4, GPT-3.5) or OpenRouter (200+ models including DeepSeek R1 FREE, Claude, Gemini, and more)
- 📚 **11+ File Formats Supported**: PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, HTML, JSON, XML, RTF
- 🚀 **Lightning-Fast RAG Pipeline**: Custom TF-IDF embeddings + ChromaDB vector search
- 💎 **Beautiful Modern UI**: Dark-themed, responsive interface with Tailwind CSS
- 🔒 **Privacy-First**: API keys stored locally in your browser, never on our servers
- ⚡ **Smart Caching**: 10-minute response cache for faster interactions
- 🎨 **Multiple Task Types**: Q&A, Summarization, Action Plans, Creative Writing

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Frontend (HTML/JS/Tailwind)                │
│  • Provider Selection (OpenAI/OpenRouter)                     │
│  • File Upload & Text Input                                   │
│  • Real-time Chat Interface                                   │
│  • API Key Management                                          │
└────────────────────┬─────────────────────────────────────────┘
                     │ REST API
┌────────────────────▼─────────────────────────────────────────┐
│                    FastAPI Backend                             │
│  • Request Validation (Pydantic)                               │
│  • Multi-Provider LLM Support                                  │
│  • File Processing Pipeline                                    │
│  • Response Caching                                            │
└────────────────────┬─────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
┌────────▼────────┐    ┌─────────▼──────────┐
│   ChromaDB      │    │  LLM Providers      │
│ Vector Database │    │  • OpenAI API       │
│ (TF-IDF)        │    │  • OpenRouter API   │
└─────────────────┘    └────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **API Key** from either:
  - [OpenAI](https://platform.openai.com/api-keys) - For GPT models
  - [OpenRouter](https://openrouter.ai/) - For 200+ models (FREE tier available)

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

3. **Run the application**
   ```bash
   python main.py
   ```

   Or use uvicorn directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 7860
   ```

4. **Access the web interface**
   Open your browser and navigate to:
   ```
   http://localhost:7860
   ```

5. **Configure your AI provider**
   - Choose between **OpenAI** or **OpenRouter** in the UI
   - Enter your API key
   - Test and save the key locally

---

## 📖 How to Use

### 1. Choose Your AI Provider

- **OpenAI**: Access to GPT-4o, GPT-4o-mini, GPT-4, GPT-3.5-turbo
- **OpenRouter**: 200+ models including DeepSeek R1 (FREE), Claude, GPT-4, Gemini, Llama 3, and more
  - **Default model**: DeepSeek R1 (completely free to use)

### 2. Upload Your Documents

ContextIQ supports a wide range of file formats:

| Category | Formats |
|----------|---------|
| **Text** | .txt, .md, .rtf |
| **Documents** | .pdf, .docx |
| **Presentations** | .pptx |
| **Data** | .xlsx, .csv, .json, .xml |
| **Web** | .html, .htm |

### 3. Index Your Content

Click "Index Context" to process and store your documents in the vector database. The system will:
- Extract text from your documents
- Split into manageable chunks (600 characters)
- Generate TF-IDF embeddings
- Store in ChromaDB for fast retrieval

### 4. Interact with Your AI Assistant

Choose from multiple task types:

- **Question & Answer**: Get precise answers from your documents
- **Summarize**: Generate concise summaries
- **Generate Action Plan**: Create actionable plans from your content
- **Creative Writing**: Transform your ideas into creative content

---

## 🎯 Features in Detail

### 📁 Advanced File Processing

Our robust file processing pipeline handles:

- **PDF**: Multi-page extraction with PyMuPDF
- **Word Documents**: Paragraphs and tables extraction
- **PowerPoint**: Slide-by-slide text extraction
- **Excel/CSV**: Structured data processing with Pandas
- **HTML**: Clean text extraction with BeautifulSoup
- **JSON/XML**: Intelligent parsing and formatting

### 🧠 Intelligent RAG Pipeline

1. **Custom TF-IDF Embeddings**
   - 384-dimensional vectors
   - N-gram support (1-2)
   - English stop words filtering
   - Fallback hashing mechanism

2. **ChromaDB Vector Database**
   - In-memory storage for speed
   - Similarity-based retrieval
   - Configurable chunk retrieval (default: 3)

3. **Smart Context Assembly**
   - Retrieves relevant chunks
   - Constructs optimized prompts
   - Respects token limits per task type

### 🔧 Configurable Settings

| Setting | Default | Description |
|---------|---------|-------------|
| MAX_TOKENS_CHAT | 4000 | Q&A response tokens |
| MAX_TOKENS_SUMMARIZE | 3000 | Summary tokens |
| MAX_TOKENS_PLAN | 5000 | Action plan tokens |
| MAX_TOKENS_CREATIVE | 6000 | Creative writing tokens |
| MAX_CHUNKS_RETRIEVE | 3 | Vector search results |
| CACHE_EXPIRATION | 600s | Response cache duration |

---

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern, fast web framework
- **ChromaDB** - Vector database for embeddings
- **Scikit-learn** - TF-IDF vectorization
- **Pydantic** - Data validation
- **OpenAI SDK** - GPT models integration
- **Requests** - HTTP client for OpenRouter

### Frontend
- **Tailwind CSS** - Utility-first CSS framework
- **Marked.js** - Markdown rendering
- **Vanilla JavaScript** - No framework bloat
- **LocalStorage** - Client-side API key storage

### File Processing
- **PyMuPDF (fitz)** - PDF processing
- **python-docx** - Word documents
- **python-pptx** - PowerPoint files
- **Pandas** - Excel/CSV handling
- **BeautifulSoup** - HTML parsing
- **striprtf** - RTF file support

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve main interface |
| `/health` | GET | Health check |
| `/api/v1/test-api-key` | POST | Validate API key |
| `/api/v1/index` | POST | Index text context |
| `/api/v1/index-file` | POST | Upload & index file |
| `/api/v1/generate` | POST | Generate AI response |
| `/api/v1/task` | POST | Execute specialized task |
| `/api/v1/clear_index` | POST | Clear vector database |

---

## 🔒 Privacy & Security

- ✅ API keys stored **only** in browser LocalStorage
- ✅ No server-side API key storage
- ✅ All requests use user-provided keys
- ✅ HTTPS recommended for production
- ✅ No telemetry or tracking
- ✅ Open source - audit the code yourself

---

## 🚢 Deployment

### Docker

```bash
docker build -t contextiq .
docker run -p 7860:7860 contextiq
```

### Hugging Face Spaces

This project is optimized for Hugging Face Spaces deployment. Simply:

1. Create a new Space
2. Upload the repository files
3. Set Space SDK to "Docker"
4. Deploy!

[View Live Demo](https://huggingface.co/spaces/Ab-Romia/Context-Aware-AI)

---

## 🎨 UI Features

- 🌙 **Dark Theme**: Easy on the eyes
- 📱 **Fully Responsive**: Works on mobile, tablet, and desktop
- 🎭 **Glass-morphism Effects**: Modern, elegant design
- ⚡ **Real-time Updates**: Live status indicators
- 📊 **Character/Word Counters**: Track your content
- 🔄 **Collapsible Sections**: Clean, organized interface
- 💬 **Markdown Support**: Rich text formatting in responses

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenRouter** for providing access to 200+ AI models
- **OpenAI** for GPT models
- **ChromaDB** for the vector database
- **FastAPI** for the amazing web framework
- **Tailwind CSS** for the beautiful UI

---

## 📬 Contact

**Ab-Romia** - Abdelrahman Abouroumia

- GitHub: [@Ab-Romia](https://github.com/Ab-Romia)
- Hugging Face: [Ab-Romia](https://huggingface.co/Ab-Romia)

---

<div align="center">

**⭐ Star this repo if you find it helpful! ⭐**

Made with ❤️ by Ab-Romia

</div>
