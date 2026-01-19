import os
import textwrap
from typing import Dict, Any, List

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Doc QnA Chatbot", page_icon="📑", layout="wide")

# Styling: purposeful typography and gradient background
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Space Grotesk', sans-serif;
    }
    .appview-container {
        background: radial-gradient(circle at 20% 20%, #0ea5e9 0, transparent 25%),
                    radial-gradient(circle at 80% 0%, #22d3ee 0, transparent 20%),
                    linear-gradient(120deg, #0f172a 0%, #0b1221 60%, #0f172a 100%);
    }
    .answer-card {background: #0b1324; padding: 1rem 1.25rem; border-radius: 14px; border: 1px solid #1e293b;}
    .pill {display: inline-block; padding: 0.25rem 0.65rem; border-radius: 999px; background: #0ea5e91a; color: #67e8f9; border: 1px solid #0ea5e9;}
    </style>
    """,
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []
if "contexts" not in st.session_state:
    st.session_state.contexts = []


def call_api(path: str, **kwargs) -> Dict[str, Any]:
    url = f"{API_BASE}{path}"
    resp = requests.request(**kwargs, url=url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def render_contexts(contexts: List[Dict[str, Any]]) -> None:
    if not contexts:
        st.info("No supporting context returned yet.")
        return
    st.subheader("Supporting context")
    for ctx in contexts:
        with st.expander(f"{ctx.get('source', 'document')} • chunk {ctx.get('chunk', '?')} • score {ctx.get('score', 0):.4f}"):
            st.write(textwrap.shorten(ctx.get("text", ""), width=1000, placeholder=" ..."))


st.title("Doc QnA Chatbot")
st.caption("Advanced retrieval-augmented generation with production-ready backend.")

with st.sidebar:
    st.header("Backend")
    api_override = st.text_input("API base", API_BASE, help="e.g. http://localhost:8000")
    if api_override:
        API_BASE = api_override.rstrip("/")

    if st.button("Health check"):
        try:
            health = call_api("/health", method="GET")
            st.success(f"Backend ready • top_k={health.get('top_k')}")
        except Exception as exc:
            st.error(f"Health check failed: {exc}")

    st.divider()
    st.header("Upload document")
    upload = st.file_uploader("Choose a PDF or text file", type=["pdf", "txt", "md"])
    if upload and st.button("Ingest file"):
        files = {"file": (upload.name, upload, upload.type)}
        try:
            ingest_resp = call_api("/ingest", method="POST", files=files)
            st.success(f"Ingested {ingest_resp.get('chunks_indexed')} chunks from {ingest_resp.get('file')}")
        except Exception as exc:
            st.error(f"Ingest failed: {exc}")

left, right = st.columns([2, 1])

with left:
    st.subheader("Ask a question")
    question = st.text_input("Your question", placeholder="What does the document say about privacy?")
    if st.button("Submit", type="primary") and question:
        try:
            response = call_api("/query", method="POST", params={"question": question})
            answer = response.get("answer", "No answer returned.")
            st.session_state.history.insert(0, {"question": question, "answer": answer})
            st.session_state.contexts = response.get("contexts", [])
        except Exception as exc:
            st.error(f"Query failed: {exc}")

    for item in st.session_state.history:
        st.markdown(
            f"""
            <div class="answer-card">
                <div class="pill">Answer</div>
                <h4>{item['question']}</h4>
                <p>{item['answer']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with right:
    render_contexts(st.session_state.contexts)
