import os
from typing import List, Dict, Any
from functools import lru_cache
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

load_dotenv()

def format_docs(docs: List[Document]) -> str:
    """Joins doc.page_content with '\\n\\n---\\n\\n'."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

@lru_cache(maxsize=4)
def get_groq_llm(model_name: str = "llama-3.3-70b-versatile") -> ChatGroq:
    """Cached ChatGroq client factory."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    return ChatGroq(
        model=model_name,
        groq_api_key=api_key,
        temperature=0
    )

def run_rag(query: str, retriever, chunk_label: str, retrieval_label: str) -> Dict[str, Any]:
    """
    Runs RAG pipeline for a given query, retriever, chunk_label, and retrieval_label.
    Returns dict with keys: chunk_method, retrieval_method, answer, and docs_retrieved.
    """
    # 1. Retrieve documents and count them
    docs = retriever.invoke(query)
    num_docs_retrieved = len(docs)

    # 2. Exact Prompt Template required
    template = "You are an AI assistant. Use ONLY the context below to answer.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    prompt = PromptTemplate.from_template(template)

    # Active supported Groq models (No decommissioned/preview models)
    candidate_models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile"
    ]

    answer = None
    last_error = None

    for model_name in candidate_models:
        try:
            llm = get_groq_llm(model_name)
            
            # 3. Chain: retriever | format_docs | prompt | llm | StrOutputParser()
            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            answer = chain.invoke(query)
            break
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if any(k in err_str for k in ["decommissioned", "model_not_found", "404", "does not exist", "model_decommissioned"]):
                continue
            else:
                raise e

    if answer is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to generate answer from Groq LLM.")

    return {
        "chunk_method": chunk_label,
        "retrieval_method": retrieval_label,
        "answer": answer,
        "docs_retrieved": num_docs_retrieved
    }
