# RAG Chatbot — AI Season Bootcamp (Task 3)

A production-grade Retrieval-Augmented Generation (RAG) system built to benchmark and compare **3 Chunking Strategies** against **2 Retrieval Methods** simultaneously across 6 combinations using **LangChain**, **ChromaDB**, **HuggingFace Embeddings**, **Groq LLM**, and **Streamlit**.

---

## 🏗️ Project Architecture

```text
RAG chatbot/
├── app/
│   ├── chunkers.py     # Defines all 3 text splitters & collection mappings
│   ├── ingest.py       # Document loader & ChromaDB vector database indexer
│   ├── retrievers.py   # Vector Similarity & MMR retriever factory
│   ├── rag.py          # Groq LLM integration & LCEL RAG chain pipeline
│   └── main.py         # Streamlit 2-column comparison UI
├── .env.example        # Environment variable template for GROQ_API_KEY
├── aiseason-document.txt # Knowledge base document (AI Season Bootcamp content)
├── requirements.txt    # Project dependencies
├── README.md           # End-to-end documentation & setup guide
└── .gitignore          # Excluded environment and database files
```

---

## ⚡ 6 RAG Strategy Combinations Benchmark

The application runs and displays output for all 6 combinations side-by-side for every query:

| Combination # | Chunking Strategy          | Retrieval Method      | ChromaDB Collection | Parameters                                      |
| :------------ | :------------------------- | :-------------------- | :------------------ | :---------------------------------------------- |
| **1**         | **Fixed Size (500)**       | **Similarity Search** | `fixed_chunks`      | `chunk_size=500, overlap=50, k=3`               |
| **2**         | **Fixed Size (500)**       | **MMR**               | `fixed_chunks`      | `chunk_size=500, overlap=50, k=3, fetch_k=10`   |
| **3**         | **Sentence-Based (200)**   | **Similarity Search** | `sentence_chunks`   | `chunk_size=200, overlap=20, k=3`               |
| **4**         | **Sentence-Based (200)**   | **MMR**               | `sentence_chunks`   | `chunk_size=200, overlap=20, k=3, fetch_k=10`   |
| **5**         | **Recursive Large (1000)** | **Similarity Search** | `recursive_chunks`  | `chunk_size=1000, overlap=100, k=3`             |
| **6**         | **Recursive Large (1000)** | **MMR**               | `recursive_chunks`  | `chunk_size=1000, overlap=100, k=3, fetch_k=10` |

---

## 🚀 End-to-End Setup & Execution Guide

### Step 1: Install Dependencies

Open your terminal in the project root directory and run:

```bash
pip install -r requirements.txt
```

_(If you have multiple Python interpreters installed on Windows, use your target Python path, e.g.:)_

```powershell
C:\Users\HD\AppData\Local\Python\pythoncore-3.14-64\python.exe -m pip install -r requirements.txt
```

### Step 2: Configure Environment Variables

Copy `.env.example` to `.env` and add your Groq API key:

```bash
cp .env.example .env
```

Edit `.env` and paste your key:

```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

### Step 3: Run Knowledge Base Ingestion

Ingest `aiseason-document.txt` from project root and build all 3 ChromaDB collections in `./chroma_db`:

```bash
python -m app.ingest
```

_(Explicit Python 3.14 command for Windows PowerShell:)_

```powershell
C:\Users\HD\AppData\Local\Python\pythoncore-3.14-64\python.exe -m app.ingest
```

**Expected Output:**

```text
Loading document from 'aiseason-document.txt'...
Fixed Size (500): 154 chunks created.
Sentence-Based (200): 357 chunks created.
Recursive Large (1000): 69 chunks created.
Ingestion completed successfully for all 3 chunking methods.
```

### Step 4: Launch the Streamlit Comparison UI

Start the web dashboard:

```bash
streamlit run app/main.py
```

_(Explicit Python 3.14 command for Windows PowerShell:)_

```powershell
C:\Users\HD\AppData\Local\Python\pythoncore-3.14-64\python.exe -m streamlit run app/main.py
```

---

## 🛠️ Resolving IDE Import Errors (`ModuleNotFoundError`)

If your IDE (VSCode / PyCharm) reports `Cannot find module langchain_community` or `streamlit`:

1. **Cause**: The IDE's default Python interpreter is pointed to an environment (such as MSYS2 Python at `C:\msys64\ucrt64\bin\python.exe`) rather than the Python installation where `pip` installed the project packages (`C:\Users\HD\AppData\Local\Python\pythoncore-3.14-64\python.exe`).
2. **Fix**: In VSCode, press `Ctrl + Shift + P` -> Select **Python: Select Interpreter** -> Choose `Python 3.14.0 (pythoncore-3.14-64)` (`C:\Users\HD\AppData\Local\Python\pythoncore-3.14-64\python.exe`).

---

## 🔧 Technical Details & Components

- **Document Loader**: `TextLoader` with UTF-8 encoding reading `aiseason-document.txt`.
- **Embedding Model**: `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` from `langchain_huggingface`.
- **Vector Database**: `Chroma` vector store stored locally at `./chroma_db`.
- **LLM Engine**: `ChatGroq` (`model="llama-3.1-8b-instant"`, `temperature=0`) with fallback support.
- **RAG Pipeline**: Built using LangChain Expression Language (LCEL):
  `{"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()`
