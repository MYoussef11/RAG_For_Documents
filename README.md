# Local RAG System for PDF Document Q&A

This project is a **local Retrieval-Augmented Generation (RAG)** system designed to answer questions from uploaded **PDF documents** — fully offline, with **no internet access required** after setup.

It uses:
- **Gradio** for a simple web UI.
- **LangChain** to build the RAG pipeline.
- **DeepSeek** or other local-compatible LLMs for answering questions.
- **ChromaDB** for storing and retrieving document embeddings.

---

## Features

-  Upload PDF documents and ask questions about their content.
-  Runs **entirely locally** — no API keys, no external LLM calls.
-  Automatically splits, indexes, and queries your PDFs.
-  Simple and clean Gradio interface.
-  Fast question answering via vector similarity + local LLM.

---

##  Quickstart

### 1. Clone the repository

```bash```
git clone https://github.com/MYoussef11/RAG_For_Documents.git
cd ~/RAG_For_Documents

### 2. Create a virtual environment and activate it

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Run the app

python app.py

### 5.1. Access Interface

A Gradio web interface will automatically open in your browser (usually at http://127.0.0.1:7860).
Upload your PDF and start asking questions based on its content.

### 5.2. Optional: Share Public Link (if needed)

interface.launch(share=True)
⚠️ Note: This requires internet access and may not work reliably inside Jupyter or VS Code.

### 6. How It Works (Under the Hood)

1.Upload: User uploads a PDF file.
2.Text Extraction: File is loaded and split into chunks using LangChain's TextSplitter.
3.Embeddings: Chunks are embedded using a local embedding model.
4.Storage: Embeddings are saved in ChromaDB (local vector store).
5.User Input: User submits a question.
6.Retrieval: System retrieves relevant chunks using vector similarity.
7.Answer Generation: The selected chunks + question are passed to an LLM (e.g., DeepSeek) to generate a final answer.
8.Display: The answer is shown in the Gradio interface.
