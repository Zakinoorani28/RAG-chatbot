import os
from typing import Dict
from functools import lru_cache
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.retrievers import BaseRetriever

load_dotenv()

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIR = os.path.abspath("./chroma_db")

@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Cached singleton for HuggingFaceEmbeddings to avoid reloading model weights."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_retrievers(collection_name: str, persist_directory: str = PERSIST_DIR) -> Dict[str, BaseRetriever]:
    """
    Loads ChromaDB collection using cached embeddings and returns two retrieval methods:
    - 'Similarity Search': vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    - 'MMR': vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 3, 'fetch_k': 10})
    """
    embeddings = get_embedding_model()
    
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    
    return {
        "Similarity Search": vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}
        ),
        "MMR": vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 3, "fetch_k": 10}
        )
    }
