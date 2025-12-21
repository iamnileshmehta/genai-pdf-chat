# genai-pdf-chat
# GenAI PDF Chatbot (RAG-based)

A conversational AI application that allows users to upload PDF documents and ask questions about their content.  
The system uses **Retrieval-Augmented Generation (RAG)** with conversational memory to provide accurate, context-aware answers along with source references.

---

## 🚀 Features

- 📄 Upload and query PDF documents
- 💬 Multi-turn conversational chat (context-aware)
- 🧠 Conversational memory using LangChain
- 🔍 Semantic search with FAISS vector database
- 📚 Source document references for answers
- ⚡ Optimized Streamlit performance using caching
- 🔄 Clear chat functionality

---

## 🧠 System Architecture (High Level)

1. **PDF Loader**  
   Loads PDF files and extracts text.

2. **Text Chunking**  
   Splits documents into overlapping chunks for better retrieval.

3. **Embeddings**  
   Converts text chunks into vector embeddings using HuggingFace models.

4. **Vector Store (FAISS)**  
   Stores and retrieves relevant document chunks based on semantic similarity.

5. **LLM (Pluggable)**  
   Uses Gemini / HuggingFace / local LLMs for answer generation.

6. **Conversational Retrieval Chain**  
   Combines retrieved context + chat history to generate accurate responses.

---

## 🧩 Why RAG?

Large Language Models can hallucinate or lack domain-specific knowledge.  
RAG improves reliability by:
- Retrieving relevant document context
- Grounding responses in source data
- Increasing accuracy and trustworthiness

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – UI
- **LangChain** – RAG & orchestration
- **FAISS** – Vector database
- **HuggingFace Embeddings**
- **Gemini / HuggingFace LLMs**
- **dotenv** – Environment management

---

##Usage
Upload a PDF document
Ask questions related to the document
View conversational answers with source references
Use Clear Chat to reset conversation memory

##Security Notes
API keys are managed using environment variables
.env and temporary files are excluded via .gitignore

##Key Learnings
Handling Streamlit reruns efficiently
Building conversational RAG systems
Managing LLM memory and UI state
Designing LLM-agnostic architectures
Debugging real-world LLM integration issues

#Future Improvements
Multi-PDF support
Persistent vector store
User authentication
Streaming responses
Model switching via UI

Author
Nilesh Mehta
AI-ML Engineer
Focused on practical, production-ready AI systems

⭐ If you find this project useful, feel free to star the repository.

## 📦 Installation

### 1️⃣ Clone the repository
```bash

##Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

##Install dependencies
pip install -r requirements.txt

##Setup environment variables
GOOGLE_API_KEY=your_api_key_here

##Run the Application
streamlit run app.py



##Project Structure
genai-pdf-chat/
│
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── screenshots/
│   ├── upload.png
│   └── chat.png
└── data/


