"""
app.py
------
The Streamlit chat interface. Run with:

    streamlit run app.py

Features:
  * multi-turn chat with memory
  * follow-up questions handled automatically (query rewriting in rag.py)
  * answer + citations + expandable source chunks
  * optional filters (year / file / chunk type) in the sidebar
"""

import config
import streamlit as st
import rag

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
st.title("📄 Conversational PDF Chatbot")
st.caption("Ask questions about your internal PDF documents.")

# ---------------------------------------------------------------------------
# Chat history lives in the session (cleared when the page reloads).
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Sidebar: optional filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters (optional)")
    year = st.text_input("Year (e.g. 2021)", value="")
    source_file = st.text_input("File name (e.g. 2021.pdf)", value="")
    chunk_type = st.selectbox("Chunk type", ["All", "text", "table"])

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

filters = {}
if year.strip().isdigit():
    filters["year"] = int(year.strip())
if source_file.strip():
    filters["source_file"] = source_file.strip()
if chunk_type != "All":
    filters["chunk_type"] = chunk_type

# ---------------------------------------------------------------------------
# Show the conversation so far
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("Sources"):
                for cite in msg["citations"]:
                    st.markdown(f"- {cite}")
        if msg.get("chunks"):
            with st.expander("Retrieved chunks"):
                for i, c in enumerate(msg["chunks"], start=1):
                    st.markdown(f"**[{i}]**")
                    st.text(c["text"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # pass prior turns (excluding the question we just added) as history
            history = st.session_state.messages[:-1]
            result = rag.answer(question, history=history, filters=filters)

        st.markdown(result["answer"])
        if result["citations"]:
            with st.expander("Sources"):
                for cite in result["citations"]:
                    st.markdown(f"- {cite}")
        if result["chunks"]:
            with st.expander("Retrieved chunks"):
                for i, c in enumerate(result["chunks"], start=1):
                    st.markdown(f"**[{i}]**")
                    st.text(c["text"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "citations": result["citations"],
            "chunks": result["chunks"],
        }
    )
