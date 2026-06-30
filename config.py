"""
config.py
---------
One place for every setting in the project. Everything reads from here.

Values come from the .env file (see .env.example). If a value is missing,
a sensible default is used. The defaults keep all downloaded models *inside*
the project folder, which makes the whole thing easy to copy to an offline
office laptop.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from a .env file in the project root (if present).
load_dotenv()

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

DATA_RAW = ROOT / "data" / "raw"          # put your PDFs here
DATA_PROCESSED = ROOT / "data" / "processed"  # parsed output + chunks.json
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", ROOT / "chroma_db"))

# Local model folders (so the project is portable / offline-friendly).
MODELS_DIR = ROOT / "models"
HF_HOME = Path(os.getenv("HF_HOME", MODELS_DIR / "hf"))               # HuggingFace cache
DOCLING_ARTIFACTS_PATH = Path(
    os.getenv("DOCLING_ARTIFACTS_PATH", MODELS_DIR / "docling")        # Docling models
)

# Make sure all folders exist.
for p in (DATA_RAW, DATA_PROCESSED, CHROMA_DB_PATH, HF_HOME, DOCLING_ARTIFACTS_PATH):
    p.mkdir(parents=True, exist_ok=True)

# Point HuggingFace at our local cache *before* any HF library is imported
# elsewhere. This is why every script imports config first.
os.environ.setdefault("HF_HOME", str(HF_HOME))

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

EMBEDDING_REPO = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_REPO  = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
 
EMBEDDING_DIR = MODELS_DIR / "embedding"
RERANKER_DIR  = MODELS_DIR / "reranker"
 
def _resolve(folder, repo):
    return str(folder) if folder.exists() and any(folder.iterdir()) else repo
 
EMBEDDING_MODEL = _resolve(EMBEDDING_DIR, EMBEDDING_REPO)
RERANKER_MODEL  = _resolve(RERANKER_DIR, RERANKER_REPO)
 

# Gemini (new google-genai SDK). gemini-2.5-flash is a good, cheap default.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
# The SDK reads GOOGLE_API_KEY or GEMINI_API_KEY automatically; we also expose it.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "internal_pdf_rag")

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------
INITIAL_RETRIEVAL_K = int(os.getenv("INITIAL_RETRIEVAL_K", "15"))  # candidates from ChromaDB
RERANKED_TOP_K = int(os.getenv("RERANKED_TOP_K", "5"))             # kept after re-ranking

# Chunking: roughly how many characters per text chunk before we split.
CHUNK_CHAR_LIMIT = int(os.getenv("CHUNK_CHAR_LIMIT", "1200"))

# ---------------------------------------------------------------------------
# Offline switch
# ---------------------------------------------------------------------------
# When OFFLINE=1, tell HuggingFace not to phone home. Set this on the office
# laptop AFTER you have downloaded the models once.
OFFLINE = os.getenv("OFFLINE", "0") == "1"
if OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
