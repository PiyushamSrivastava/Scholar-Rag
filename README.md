# ScholarRAG — Research Paper Intelligence

ScholarRAG is a Retrieval-Augmented Generation (RAG) app that lets you upload PDF documents, ask questions about their content, and receive answers grounded in retrieved passages.

**[🚀 Try the live app](https://scholar-rag-9qn9lcye7bqgimebaehnqc.streamlit.app/)**

## Features

- Upload and extract text from PDF documents
- Split document text into chunks for retrieval
- Generate embeddings with Google Gemini
- Store and search vectors using Qdrant Cloud
- Generate answers using retrieved document context
- Display relevant source passages alongside answers

## How It Works

1. Upload a PDF.
2. ScholarRAG extracts and chunks the document text.
3. Gemini creates embeddings for the chunks.
4. Qdrant stores the vectors and document metadata.
5. When you ask a question, ScholarRAG retrieves relevant passages.
6. Gemini generates an answer based on the retrieved context.

## Tech Stack

- Python
- Streamlit
- Google Gemini
- Qdrant Cloud
- PyPDF
- LangChain Text Splitters
- `uv`

## Run Locally

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- A Gemini API key
- A Qdrant Cloud URL and API key

### 1. Clone the repository

```powershell
git clone https://github.com/PiyushamSrivastava/Scholar-Rag.git
cd Scholar-Rag
