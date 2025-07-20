# 📚 (DocQuery Pilot)RAG PDF Chatbot

This project implements a **Retrieval-Augmented Generation (RAG)** system that allows users to **chat with the contents of PDF documents**. Upload one or more PDFs and ask questions about them — the system retrieves relevant information from the documents and provides accurate, conversational responses.

---

## 🚀 Features

- 🔍 Extracts text from PDF files
- 🤖 Enables chat-style interaction with document content
- 🔗 Uses vector embeddings and similarity search for context-aware responses
- 🔐 Integrates with Gemini API for language generation
- 📁 Supports multiple PDFs

---

## 📦 Tech Stack

- Python
- Streamlit (for frontend interface)
- LangChain
- FAISS (for vector database)
- GeminiAI 
- PyPDF  (for PDF parsing)
- dotenv (for environment variable handling)

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```
### 2. Create and Activate Virtual Environment

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

### 3. Install Requirements

pip install -r requirements.txt

### 4. Add Your API Key

GOOGLE_API_KEY=your_api_key_here

