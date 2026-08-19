from typing import Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

COLLECTION_MAP = {
    "Fixed Size (500)": "fixed_chunks",
    "Sentence-Based (200)": "sentence_chunks",
    "Recursive Large (1000)": "recursive_chunks"
}

def get_all_chunkers() -> Dict[str, RecursiveCharacterTextSplitter]:
    """
    Returns a dictionary of all 3 required chunker configurations:
    - Fixed Size (500, overlap=50)
    - Sentence-Based (200, overlap=20)
    - Recursive Large (1000, overlap=100)
    """
    return {
        "Fixed Size (500)": RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50
        ),
        "Sentence-Based (200)": RecursiveCharacterTextSplitter(
            chunk_size=200, 
            chunk_overlap=20, 
            separators=[". ", "! ", "? ", "\n"]
        ),
        "Recursive Large (1000)": RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
    }
