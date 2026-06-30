"""
rag.py
------
The brain of the chatbot. Given a question (and the chat history), it:

  1. rewrites a follow-up question into a standalone one  (using Gemini)
  2. retrieves candidate chunks from ChromaDB
  3. re-ranks them with a cross-encoder and keeps the best few
  4. asks Gemini to answer using ONLY those chunks
  5. builds citations from chunk metadata (not from the LLM, so no made-up sources)

Everything here is plain functions so it is easy to read and debug.
"""

import config
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# Models and clients are loaded once and reused (loading is slow).
_embedder = None
_reranker = None
_collection = None
_gemini = None


# ---------------------------------------------------------------------------
# Lazy loaders
# ---------------------------------------------------------------------------
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(config.RERANKER_MODEL)
    return _reranker


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DB_PATH))
        _collection = client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def get_gemini():
    global _gemini
    if _gemini is None:
        from google import genai
        if not config.GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set in your .env file.")
        _gemini = genai.Client(api_key=config.GOOGLE_API_KEY)
    return _gemini


def _ask_gemini(prompt: str) -> str:
    resp = get_gemini().models.generate_content(
        model=config.GEMINI_MODEL, contents=prompt
    )
    return (resp.text or "").strip()


# ---------------------------------------------------------------------------
# 1. Query rewriting (only when there is history)
# ---------------------------------------------------------------------------
def rewrite_query(history, question: str) -> str:
    """Turn a follow-up like 'does it change after 5 years?' into a full question."""
    if not history:
        return question

    convo = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
    prompt = (
        "Given the conversation below, rewrite the user's latest question as a "
        "standalone question that makes sense on its own. Keep it short. "
        "Return ONLY the rewritten question, nothing else.\n\n"
        f"Conversation:\n{convo}\n\n"
        f"Latest question: {question}\n\n"
        "Standalone question:"
    )
    try:
        rewritten = _ask_gemini(prompt)
        return rewritten or question
    except Exception:
        return question  # if rewriting fails, just use the original


# ---------------------------------------------------------------------------
# 2. Retrieval from ChromaDB
# ---------------------------------------------------------------------------
def _build_where(filters: dict):
    """Build a ChromaDB 'where' clause from optional metadata filters."""
    if not filters:
        return None
    active = {k: v for k, v in filters.items() if v not in (None, "", "All")}
    if not active:
        return None
    if len(active) == 1:
        k, v = next(iter(active.items()))
        return {k: v}
    return {"$and": [{k: v} for k, v in active.items()]}


def retrieve(query: str, filters: dict = None):
    embedding = get_embedder().encode([query], convert_to_numpy=True).tolist()
    results = get_collection().query(
        query_embeddings=embedding,
        n_results=config.INITIAL_RETRIEVAL_K,
        where=_build_where(filters),
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return [{"text": d, "metadata": m} for d, m in zip(docs, metas)]


# ---------------------------------------------------------------------------
# 3. Re-ranking with a cross-encoder
# ---------------------------------------------------------------------------
def rerank(query: str, candidates: list):
    if not candidates:
        return []
    pairs = [(query, c["text"]) for c in candidates]
    scores = get_reranker().predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[: config.RERANKED_TOP_K]]


# ---------------------------------------------------------------------------
# 4. Build context + 5. citations
# ---------------------------------------------------------------------------
def build_context(chunks):
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] {c['text']}")
    return "\n\n".join(parts)


def build_citations(chunks):
    cites = []
    for i, c in enumerate(chunks, start=1):
        m = c["metadata"]
        page = m.get("page_number", "?")
        section = m.get("section") or "—"
        cites.append(
            f"[{i}] {m.get('source_file', '?')} | Page {page} | "
            f"{section} | {m.get('chunk_type', 'text')}"
        )
    return cites


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def answer(question: str, history=None, filters: dict = None):
    history = history or []

    standalone = rewrite_query(history, question)
    candidates = retrieve(standalone, filters)
    top_chunks = rerank(standalone, candidates)

    if not top_chunks:
        return {
            "answer": "I could not find anything relevant in the documents.",
            "citations": [],
            "chunks": [],
            "rewritten_query": standalone,
        }

    context = build_context(top_chunks)
    prompt = (
        "You are a helpful assistant answering questions about internal PDF "
        "documents. Use ONLY the context below to answer. If the answer is not "
        "in the context, say you don't know. Be concise and refer to sources "
        "by their number like [1], [2] where useful.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    generated = _ask_gemini(prompt)

    return {
        "answer": generated,
        "citations": build_citations(top_chunks),
        "chunks": top_chunks,
        "rewritten_query": standalone,
    }
