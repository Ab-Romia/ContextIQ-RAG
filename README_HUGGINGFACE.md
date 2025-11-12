---
title: ContextIQ - Context-Aware AI Assistant
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
pinned: true
license: mit
app_port: 7860
---

# 🧠 ContextIQ - Intelligent Context-Aware AI Assistant

Welcome to **ContextIQ**, a sophisticated RAG (Retrieval-Augmented Generation) application that transforms how you interact with your documents!

## 🌟 What Can You Do?

- 📚 **Upload Documents**: Support for 11+ file formats (PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, HTML, JSON, XML, RTF)
- 🤖 **Ask Questions**: Get intelligent answers based on your uploaded documents
- 📝 **Summarize**: Generate concise summaries of your content
- 📋 **Action Plans**: Create actionable plans from your documents
- ✍️ **Creative Writing**: Transform your ideas into creative content

## 🎯 Dual AI Provider Support

Choose your preferred AI provider:

### OpenRouter (FREE DeepSeek Model!)
- 200+ models including DeepSeek R1 (FREE), Claude, GPT-4, Gemini, Llama 3
- **Default**: DeepSeek R1 - completely free to use
- Get your key: [openrouter.ai](https://openrouter.ai/)

### OpenAI
- GPT-4o, GPT-4o-mini, GPT-4, GPT-3.5-turbo
- Production-ready models
- Get your key: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

## 🚀 How to Use

1. **Choose Your AI Provider**
   - Select OpenRouter (free) or OpenAI in the interface

2. **Enter Your API Key**
   - Your key is stored locally in your browser only
   - Never sent to our servers

3. **Upload Your Documents**
   - Drag & drop or browse for files
   - Or paste text directly

4. **Index Your Content**
   - Click "Index Context" to process your documents

5. **Start Asking Questions!**
   - Choose a task type (Q&A, Summarize, Plan, Creative)
   - Type your question or prompt
   - Get AI-powered responses based on your documents

## 🔒 Privacy & Security

- ✅ Your API keys are stored **only** in your browser
- ✅ No server-side storage of API keys
- ✅ All requests use your own API key
- ✅ Open source - audit the code yourself

## 🛠️ Technology Stack

- **Backend**: FastAPI + Python
- **Vector Database**: ChromaDB with custom TF-IDF embeddings
- **Frontend**: Tailwind CSS + Vanilla JavaScript
- **AI Providers**: OpenAI SDK + OpenRouter API
- **File Processing**: PyMuPDF, python-docx, pandas, BeautifulSoup, and more

## 📊 Supported File Formats

| Category | Formats |
|----------|---------|
| **Text** | .txt, .md, .rtf |
| **Documents** | .pdf, .docx |
| **Presentations** | .pptx |
| **Data** | .xlsx, .csv, .json, .xml |
| **Web** | .html, .htm |

## 💡 Tips for Best Results

- **Clear Questions**: Ask specific questions about your documents
- **Context Matters**: The more relevant text you provide, the better the answers
- **Chunk Size**: Large documents are automatically split into manageable chunks
- **Model Selection**:
  - Use OpenRouter's DeepSeek R1 (FREE) for excellent reasoning at no cost
  - Use OpenAI's GPT-4o for production workloads
  - Default DeepSeek model is completely free - no credit card needed!

## 🤝 Open Source

This project is open source! Check out the code on GitHub:
[github.com/Ab-Romia/ContextIQ-RAG](https://github.com/Ab-Romia/ContextIQ-RAG)

## 📬 Feedback

Found a bug or have a feature request?
[Open an issue on GitHub](https://github.com/Ab-Romia/ContextIQ-RAG/issues)

---

Made with ❤️ by Ab-Romia (Abdelrahman Abouroumia)
