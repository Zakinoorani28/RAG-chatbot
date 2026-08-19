import os
from typing import Dict
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from app.chunkers import get_all_chunkers, COLLECTION_MAP

load_dotenv()

DOC_PATH = "aiseason-document.txt"
PERSIST_DIR = os.path.abspath("./chroma_db")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def load_document(file_path: str = DOC_PATH):
    """Loads the knowledge base document from project root using TextLoader."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found at root path: {file_path}")
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()

def ingest_all_collections(file_path: str = DOC_PATH, persist_dir: str = PERSIST_DIR) -> Dict[str, int]:
    """
    Splits aiseason-document.txt using 3 chunking methods and saves each 
    into a separate ChromaDB collection inside persist_directory.
    """
    print(f"Loading document from '{file_path}'...")
    documents = load_document(file_path)
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    chunkers = get_all_chunkers()
    results = {}
    
    for label, splitter in chunkers.items():
        collection_name = COLLECTION_MAP[label]
        chunks = splitter.split_documents(documents)
        results[label] = len(chunks)
        print(f"{label}: {len(chunks)} chunks created.")
        
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir
        )
        
    print("Ingestion completed successfully for all 3 chunking methods.")
    return results

if __name__ == "__main__":
    ingest_all_collections()
