import os
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from app.chunkers import COLLECTION_MAP
from app.ingest import ingest_all_collections
from app.retrievers import get_retrievers
from app.rag import run_rag

load_dotenv()

PERSIST_DIR = os.path.abspath("./chroma_db")

st.set_page_config(
    page_title="RAG Chatbot — AI Season Bootcamp",
    page_icon="🤖",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Project Overview")
    st.write("📄 **Knowledge Base:** `aiseason-document.txt`")
    st.write("🧩 **Chunking Methods:** 3")
    st.write("🔍 **Retrieval Methods:** 2")
    st.write("⚡ **Total Combinations:** 6")
    st.markdown("---")
    
    if st.button("Re-run Ingestion"):
        with st.spinner("Ingesting document and indexing collections..."):
            try:
                ingest_all_collections()
                st.success("Ingestion complete! All 3 collections indexed.")
            except Exception as e:
                st.error(f"Ingestion error: {e}")

# Check if chroma_db exists
if not os.path.exists(PERSIST_DIR):
    st.error("Run ingest.py first: python -m app.ingest")

st.title("RAG Chatbot — AI Season Bootcamp")
st.header("Ask anything about AI Season Bootcamp")

query = st.text_input("Enter your question:")
submit_btn = st.button("Submit")

if "results" not in st.session_state:
    st.session_state["results"] = []

def eval_task(task_args):
    query_str, chunk_label, coll_name, retrieval_label = task_args
    try:
        retrievers_dict = get_retrievers(coll_name)
        retriever = retrievers_dict[retrieval_label]
        return run_rag(
            query=query_str,
            retriever=retriever,
            chunk_label=chunk_label,
            retrieval_label=retrieval_label
        )
    except Exception as e:
        return {
            "chunk_method": chunk_label,
            "retrieval_method": retrieval_label,
            "answer": f"⚠️ Error: {e}",
            "docs_retrieved": 0
        }

if submit_btn and query:
    if not os.path.exists(PERSIST_DIR):
        st.error("Run ingest.py first: python -m app.ingest")
    else:
        with st.spinner("Running 6 RAG combinations in parallel..."):
            chunker_labels = [
                "Fixed Size (500)",
                "Sentence-Based (200)",
                "Recursive Large (1000)"
            ]
            retrieval_types = ["Similarity Search", "MMR"]
            
            tasks = []
            for chunk_label in chunker_labels:
                coll_name = COLLECTION_MAP[chunk_label]
                for retrieval_label in retrieval_types:
                    tasks.append((query, chunk_label, coll_name, retrieval_label))
                    
            # Run all 6 combinations in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=6) as executor:
                results = list(executor.map(eval_task, tasks))
                
            st.session_state["results"] = results

# Display stored results
if st.session_state["results"]:
    st.markdown("---")
    cols = st.columns(2)
    
    for idx, res in enumerate(st.session_state["results"]):
        with cols[idx % 2]:
            with st.container():
                st.subheader(f"Chunk: {res['chunk_method']}")
                st.caption(f"Retrieval: {res['retrieval_method']}")
                st.write(res["answer"])
                st.info(f"Docs retrieved: {res['docs_retrieved']}")
                st.divider()
