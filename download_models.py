"""
download_models.py
------------------
Run this ONCE on a machine that has internet (your home laptop, or the office
laptop during first-time setup). It pulls every model the project needs into
the local ./models folder so the app can later run fully offline.

    python download_models.py

What it downloads:
  1. Docling layout / table / OCR models  -> ./models/docling
  2. HuggingFace embedding model          -> ./models/hf
  3. HuggingFace cross-encoder re-ranker   -> ./models/hf

After this finishes, you can zip the whole project folder (models included)
and copy it to an offline machine. There, set OFFLINE=1 in .env and run normally.
"""

import subprocess
import sys

import config  # sets HF_HOME etc. before HF libraries load


def download_docling_models():
    print("\n[1/3] Downloading Docling models ...")
    target = str(config.DOCLING_ARTIFACTS_PATH)

    # Preferred: the Python helper that ships with Docling.
    try:
        from docling.utils.model_downloader import download_models
        download_models(output_dir=target, progress=True)
        print(f"      Docling models saved to: {target}")
        return
    except Exception as e:  # signature changes between versions -> fall back to CLI
        print(f"      Python helper unavailable ({e}); trying the CLI ...")

    # Fallback: the command-line tool installed with docling.
    try:
        subprocess.run(
            ["docling-tools", "models", "download", "-o", target],
            check=True,
        )
        print(f"      Docling models saved to: {target}")
    except Exception as e:
        print(f"      Could not download Docling models automatically: {e}")
        print("      You can run it manually:")
        print(f"         docling-tools models download -o {target}")


def download_hf_model(name: str):
    """Download a sentence-transformers / cross-encoder model into HF_HOME."""
    from huggingface_hub import snapshot_download
    print(f"      pulling {name} ...")
    snapshot_download(repo_id=name)  # lands in config.HF_HOME (set in config.py)


def download_hf_models():
    print("\n[2/3] Downloading embedding model ...")
    download_hf_model(config.EMBEDDING_MODEL)

    print("\n[3/3] Downloading re-ranker model ...")
    download_hf_model(config.RERANKER_MODEL)
    print(f"      HuggingFace models saved under: {config.HF_HOME}")


if __name__ == "__main__":
    print("Downloading all models into ./models (this can take a few minutes)")
    download_docling_models()
    try:
        download_hf_models()
    except Exception as e:
        print(f"\nHuggingFace download failed: {e}")
        sys.exit(1)
    print("\nAll done. The project can now run offline (set OFFLINE=1 in .env).")
