# Conversational PDF RAG Chatbot

Ask questions about your internal PDF documents (policies, SOPs, manuals) in a
normal back-and-forth chat. The bot reads your PDFs, finds the relevant parts,
and answers using **only** what's in the documents — with citations so you can
check every answer.

Built with **Docling** (PDF parsing), **HuggingFace** (embeddings + re-ranking),
**ChromaDB** (vector store), **Gemini** (answers), and **Streamlit** (UI).

> Designed to run **fully offline** on a locked-down office laptop. See the
> "Offline setup" section — that's the important bit.

---

## How it works (in one breath)

```
PDFs → Docling parses them → chunks with metadata → embeddings → ChromaDB
                                                                     │
Your question → (rewrite if follow-up) → search ChromaDB → re-rank → Gemini
                                                                     │
                                              answer + citations in the UI
```

There are only a handful of scripts, each doing one job:

| File                  | What it does                                              |
|-----------------------|-----------------------------------------------------------|
| `config.py`           | All settings (reads `.env`). Everything imports this.     |
| `download_models.py`  | Pulls every model into `./models` for offline use.        |
| `ingest.py`           | PDFs → chunks (`data/processed/`).                         |
| `build_index.py`      | chunks → embeddings → ChromaDB.                           |
| `rag.py`              | retrieve → re-rank → ask Gemini → build citations.        |
| `app.py`              | The Streamlit chat interface.                             |

---

## What you need

- **Python 3.10 or newer** (3.12 recommended).
  On macOS the built-in Python is old (3.9). Install a modern one with Homebrew:
  ```bash
  brew install python@3.12
  ```
- A **Gemini API key** — free from https://aistudio.google.com/apikey
- Your **PDFs**, placed in `data/raw/`.

---

## Quick start (a machine WITH internet)

```bash
# 1. Create and activate a virtual environment
python3.12 -m venv venv
source venv/bin/activate            # macOS / Linux
# venv\Scripts\activate             # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your key
cp .env.example .env
# open .env and paste your Gemini API key into GOOGLE_API_KEY

# 4. Download the models (Docling + HuggingFace) into ./models
python download_models.py

# 5. Put PDFs in data/raw/  then build the index
python ingest.py
python build_index.py

# 6. Start the chatbot
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501) and start asking.

---

## Offline setup (the office laptop)

The whole point of `download_models.py` is that **the models live inside the
project folder** (`./models`). So you can prepare everything at home, copy it
across, and run with no internet.

**Step A — on a machine with internet** (your home laptop):

```bash
pip install -r requirements.txt
python download_models.py
```

This creates:

```
models/
├── docling/      ← Docling layout / table / OCR models
└── hf/           ← embedding + re-ranker models
```

**Step B — move the project to the office laptop.** Zip the entire folder
(including `models/`) and copy it over. If the office laptop already has the
Python packages installed but no internet, you only need to bring the `models/`
folder.

**Step C — on the office laptop**, turn on offline mode in `.env`:

```
OFFLINE=1
```

That sets `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` so HuggingFace never tries
to phone home. Docling already reads its models from `./models/docling` via the
`artifacts_path` setting in `config.py`. Then run the app normally:

```bash
python ingest.py
python build_index.py
streamlit run app.py
```

> **Manual Docling download** (if the automatic step ever fails):
> ```bash
> docling-tools models download -o ./models/docling
> ```
> The `artifacts_path` must point at the **parent** folder that contains the
> individual model sub-folders — which is exactly what `./models/docling` is.

> **Note on Gemini and offline:** the embedding, re-ranking, and PDF parsing all
> run locally with no internet. Only the **final answer** uses the Gemini API,
> which needs a network connection. If your office laptop has zero internet, you
> can still parse, index, and retrieve — you'd just need to swap Gemini for a
> local LLM to get generated answers (see "Swapping the LLM" below).

---

## Configuration (`.env`)

| Variable                 | Default                                          | Meaning                                  |
|--------------------------|--------------------------------------------------|------------------------------------------|
| `GOOGLE_API_KEY`         | —                                                | Your Gemini key (required for answers).  |
| `GEMINI_MODEL`           | `gemini-2.5-flash`                               | Which Gemini model to use.               |
| `EMBEDDING_MODEL`        | `sentence-transformers/all-MiniLM-L6-v2`         | Turns text into vectors.                 |
| `RERANKER_MODEL`         | `cross-encoder/ms-marco-MiniLM-L-6-v2`           | Re-orders results by relevance.          |
| `INITIAL_RETRIEVAL_K`    | `15`                                             | Candidates pulled from ChromaDB.         |
| `RERANKED_TOP_K`         | `5`                                              | Kept after re-ranking.                   |
| `CHUNK_CHAR_LIMIT`       | `1200`                                           | Rough max size of a text chunk.          |
| `OFFLINE`                | `0`                                              | Set to `1` to forbid internet model pulls.|

---

## Good to know

- **Docling is the primary parser.** If Docling isn't installed or its models
  aren't downloaded yet, `ingest.py` automatically falls back to
  **pdfplumber** (text + tables) and **PyMuPDF** (text). So the pipeline still
  runs while you sort out Docling. Docling gives the best quality (reading order,
  sections, clean tables), so download its models for real use.
- **Citations come from chunk metadata, not the LLM**, so they can't be made up.
- **Scanned PDFs:** open `ingest.py` and set `pipeline_options.do_ocr = True`
  in `parse_with_docling`. (OCR models are included in the Docling download.)
- **Re-indexing is safe.** Chunk IDs are content hashes, so re-running
  `ingest.py` / `build_index.py` won't create duplicates.

### Swapping the LLM

Everything LLM-related is in `rag.py` inside `_ask_gemini` and `get_gemini`.
To use a different provider or a local model, change just those two functions —
the rest of the pipeline doesn't care where the answer text comes from.

---

## Troubleshooting

- **"GOOGLE_API_KEY is not set"** — copy `.env.example` to `.env` and paste your
  key.
- **Docling download is slow / fails** — run the manual command above, or just
  rely on the pdfplumber fallback for now.
- **"No PDFs found"** — put your `.pdf` files in `data/raw/`.
- **Old macOS Python errors** — make sure you created the venv with
  `python3.12`, not the system `python3`.
- **The new Gemini SDK** is `google-genai` (`from google import genai`). The old
  `google-generativeai` package is deprecated — don't install it.
