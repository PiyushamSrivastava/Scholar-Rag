# ScholarRAG — Research Paper Intelligence

ScholarRAG is a Retrieval-Augmented Generation (RAG) application that lets users upload PDF documents, ask questions about their content, and receive answers grounded in retrieved passages.

**[🚀 Try the live app](https://scholar-rag-9qn9lcye7bqgimebaehnqc.streamlit.app/)**

## Features

- Upload PDF documents and extract their text
- Split extracted text into chunks for retrieval
- Generate embeddings using Google Gemini
- Store and search vectors using Qdrant Cloud
- Generate answers using retrieved document context
- Display relevant source passages alongside answers
- Indicate when relevant information is not found in the retrieved content

## How It Works

1. Upload a PDF document.
2. ScholarRAG extracts the text and splits it into chunks.
3. Gemini generates embeddings for the chunks.
4. Qdrant stores the vectors and document metadata.
5. When a question is submitted, ScholarRAG retrieves relevant passages.
6. Gemini generates an answer based on the retrieved context.

## Tech Stack

| Component | Technology |
|---|---|
| User interface | Streamlit |
| Language | Python |
| Embeddings and answer generation | Google Gemini |
| Vector database | Qdrant Cloud |
| PDF text extraction | PyPDF |
| Text splitting | LangChain Text Splitters |
| Dependency management | `uv` |

## Run Locally

### Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- A Gemini API key
- A Qdrant Cloud URL and API key

### 1. Clone the repository

```powershell
git clone https://github.com/PiyushamSrivastava/Scholar-Rag.git
cd Scholar-Rag
```

### 2. Install dependencies

```powershell
uv sync
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

Replace the example values with your own credentials. Keep `.env` private and never commit API keys or other secrets.

### 4. Run the app

```powershell
uv run streamlit run src/scholarrag/app.py
```

## Deployment

ScholarRAG is deployed on Streamlit Community Cloud.

**[Open the live app](https://scholar-rag-9qn9lcye7bqgimebaehnqc.streamlit.app/)**

Configure the following secrets in the app’s Streamlit Cloud settings:

- `GEMINI_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`

Do not store credentials in the GitHub repository.

## Project Structure

```text
Scholar-Rag/
├── src/
│   └── scholarrag/
│       ├── app.py
│       ├── ingestion.py
│       ├── chunking.py
│       ├── embeddings.py
│       ├── qdrant_db.py
│       ├── vector_store.py
│       ├── retrieval.py
│       └── rag_pipeline.py
├── data/
│   └── raw/              # Local PDFs; excluded from Git
├── .env                  # Local secrets; excluded from Git
├── .gitignore
├── .python-version
├── pyproject.toml
└── uv.lock
```

## Limitations

- Works with text-based PDFs; scanned or image-only PDFs may require OCR.
- Answer quality depends on the text extracted and the passages retrieved.
- Uploaded files on the hosted app may not persist after the hosting environment restarts.

## Future Improvements

- OCR support for scanned PDFs
- Improved document management
- More detailed page-level source references
- Retrieval and answer-quality evaluation
- Persistent document storage

## Author

**Piyusham Srivastava**

- GitHub: [@PiyushamSrivastava](https://github.com/PiyushamSrivastava)

---

Built as a hands-on project to explore document ingestion, semantic retrieval, vector databases, and Retrieval-Augmented Generation.
