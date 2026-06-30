"""
build_index.py
--------------
Step 2 of the pipeline. Reads data/processed/merged_chunks.json, turns each
chunk into a vector with a HuggingFace embedding model, and stores everything
(text + vector + metadata) in a local ChromaDB database.

Run:  python build_index.py
"""

import json

import config
import chromadb
from sentence_transformers import SentenceTransformer


def sanitize_metadata(meta: dict) -> dict:
    """ChromaDB only accepts str/int/float/bool. Replace None with ''."""
    clean = {}
    for k, v in meta.items():
        clean[k] = "" if v is None else v
    return clean


def main():
    merged = config.DATA_PROCESSED / "merged_chunks.json"
    if not merged.exists():
        print("merged_chunks.json not found. Run `python ingest.py` first.")
        return

    chunks = json.loads(merged.read_text())
    if not chunks:
        print("No chunks to index.")
        return
    print(f"Loaded {len(chunks)} chunks")

    # 1. Load the embedding model (downloaded by download_models.py).
    print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [sanitize_metadata(c["metadata"]) for c in chunks]

    print("Computing embeddings ...")
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    ).tolist()

    # 2. Store in ChromaDB. We pass embeddings ourselves so Chroma never tries
    #    to download its own model (important for offline use).
    client = chromadb.PersistentClient(path=str(config.CHROMA_DB_PATH))
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Writing to ChromaDB collection '{config.CHROMA_COLLECTION_NAME}' ...")
    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Done. Collection now holds {collection.count()} chunks.")


if __name__ == "__main__":
    main()
