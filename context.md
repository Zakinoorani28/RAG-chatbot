# Project Context & Progress Log

## Workspace

`RAG chatbot` - Knowledge Base Ingestion & Multi-Strategy Vector Indexing.

---

## Action Log

### 1. Ingestion Pipeline (`app/ingest.py`)

- **Source Document**: `aiseason-document.txt` loaded via `TextLoader` with `encoding="utf-8"`.
- **Embedding Model**: `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` from `langchain_huggingface`.
- **Vector DB Directory**: Normalized to absolute path `os.path.abspath("./chroma_db")`.
- **Collections & Chunkers**:
  1. **`Fixed Size (500)`** -> `fixed_chunks` (`chunk_size=500, chunk_overlap=50`): **154 chunks** indexed.
  2. **`Sentence-Based (200)`** -> `sentence_chunks` (`chunk_size=200, chunk_overlap=20`, `separators=['. ', '! ', '? ', '\n']`): **357 chunks** indexed.
  3. **`Recursive Large (1000)`** -> `recursive_chunks` (`chunk_size=1000, chunk_overlap=100`): **69 chunks** indexed.

### 2. Retrievers Module (`app/retrievers.py`)

- **Function**: `get_retrievers(collection_name: str) -> dict`
- **Vector DB Directory**: `os.path.abspath("./chroma_db")`.
- **Embedding Optimization**: Implemented `@lru_cache` singleton factory `get_embedding_model()`.
- **Embedding Model**: `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` from `langchain_huggingface`.
- **Retrieval Strategies**:
  1. **`Similarity Search`**: `vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})`.
  2. **`MMR`**: `vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})`.

### 3. RAG Chain Module (`app/rag.py`)

- **Active Model List**: `llama-3.1-8b-instant` (primary fast model) and `llama-3.3-70b-versatile` (intelligent flagship model). Purged `llama-3.2-3b-preview`, `gemma2-9b-it`, and `llama3-8b-8192`.
- **Prompt**: `"You are an AI assistant. Use ONLY the context below to answer.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"`
- **Helper**: `format_docs(docs)` joining page content with `"\n\n---\n\n"`.
- **Function**: `run_rag(query, retriever, chunk_label, retrieval_label) -> dict`.

### 4. Parallel Streamlit Comparison UI (`app/main.py`)

- **Vector DB Path**: `os.path.abspath("./chroma_db")`.
- **Parallel Execution Engine**: `concurrent.futures.ThreadPoolExecutor(max_workers=6)` for ~1.5s multi-strategy RAG response times.
- **Page Title**: `"RAG Chatbot — AI Season Bootcamp"` (icon: `🤖`).
- **Sidebar**: Overview stats and `"Re-run Ingestion"` button.
- **Validation**: Checks if `os.path.abspath("./chroma_db")` exists.

### 5. Setup & Documentation (`README.md` & `.gitignore`)

- **`README.md`**: Setup steps (`pip install -r requirements.txt`, `cp .env.example .env`, `python -m app.ingest`, `streamlit run app/main.py`).
- **`.gitignore`**: Excludes `.env`, virtual environments, caches.
