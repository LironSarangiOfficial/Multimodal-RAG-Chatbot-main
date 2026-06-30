"""
ingest.py
---------
Reads every PDF in data/raw/, turns it into clean, metadata-rich chunks, and
writes them to data/processed/. This is step 1 of the pipeline.

Parsing strategy (best tool for each situation):
  * Docling     -> primary parser. Layout-aware, understands reading order,
                   headings, and tables. Best quality.
  * pdfplumber  -> fallback for text AND tables if Docling is unavailable.
  * PyMuPDF     -> secondary text fallback when pdfplumber returns nothing.

Each chunk looks like:
  {
    "id": "<md5 hash>",
    "text": "...",
    "metadata": {
        "source_file": "2021.pdf",
        "year": 2021,
        "page_number": 18,
        "section": "Annual Leave",
        "chunk_type": "text" | "table"
    }
  }

Run:  python ingest.py
"""

import hashlib
import json
import re

import config


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def detect_year(filename: str):
    """Pull a 4-digit year out of a filename like '2021.pdf'. None if absent."""
    m = re.search(r"(19|20)\d{2}", filename)
    return int(m.group(0)) if m else None


def make_chunk_id(source_file: str, page, idx: int, text: str) -> str:
    """Stable content hash so re-running ingestion does not create duplicates."""
    raw = f"{source_file}|{page}|{idx}|{text[:80]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def rows_to_markdown(rows) -> str:
    """Turn a list-of-lists table into a simple markdown table (pdfplumber path)."""
    rows = [[(c if c is not None else "") for c in row] for row in rows if row]
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    lines = ["| " + " | ".join(str(c) for c in header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for r in body:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def build_chunk(text, source_file, page, section, chunk_type, idx):
    text = (text or "").strip()
    if not text:
        return None
    return {
        "id": make_chunk_id(source_file, page, idx, text),
        "text": text,
        "metadata": {
            "source_file": source_file,
            "year": detect_year(source_file),
            "page_number": page,
            "section": section or "",
            "chunk_type": chunk_type,
        },
    }


# ---------------------------------------------------------------------------
# Parser 1: Docling (primary)
# ---------------------------------------------------------------------------
def parse_with_docling(pdf_path):
    """Return a list of raw items: (text, page, section, chunk_type). Raises on failure."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    pipeline_options = PdfPipelineOptions(
        artifacts_path=str(config.DOCLING_ARTIFACTS_PATH)  # use local models
    )
    pipeline_options.do_table_structure = True
    pipeline_options.do_ocr = False  # set True for scanned PDFs

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    result = converter.convert(str(pdf_path))
    doc = result.document

    items = []
    current_section = ""
    for item, _level in doc.iterate_items():
        label = str(getattr(item, "label", "")).lower()

        # page number lives in the item's provenance, if present
        page = None
        prov = getattr(item, "prov", None)
        if prov:
            page = getattr(prov[0], "page_no", None)

        # tables -> markdown, kept as their own chunk
        if "table" in label and hasattr(item, "export_to_markdown"):
            try:
                md = item.export_to_markdown()
            except Exception:
                md = ""
            if md.strip():
                items.append((md, page, current_section, "table"))
            continue

        text = getattr(item, "text", "") or ""
        if not text.strip():
            continue

        # headings update the current section AND stay searchable as text
        if "section_header" in label or label in ("title", "section_header"):
            current_section = text.strip()

        items.append((text, page, current_section, "text"))

    if not items:
        raise RuntimeError("Docling returned no content")
    return items


# ---------------------------------------------------------------------------
# Parser 2: pdfplumber + PyMuPDF (fallback)
# ---------------------------------------------------------------------------
def parse_with_fallback(pdf_path):
    """Return a list of raw items using pdfplumber (text + tables) and PyMuPDF."""
    import pdfplumber
    import fitz  # PyMuPDF

    items = []
    fitz_doc = fitz.open(str(pdf_path))

    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            page_no = i + 1

            # text: pdfplumber first, PyMuPDF as a safety net
            text = page.extract_text() or ""
            if not text.strip() and i < fitz_doc.page_count:
                text = fitz_doc[i].get_text() or ""
            if text.strip():
                items.append((text, page_no, "", "text"))

            # tables: pdfplumber is good at these
            for table in page.extract_tables() or []:
                md = rows_to_markdown(table)
                if md.strip():
                    items.append((md, page_no, "", "table"))

    fitz_doc.close()
    if not items:
        raise RuntimeError("Fallback parser returned no content")
    return items


# ---------------------------------------------------------------------------
# Turn raw items into final chunks (merge small text pieces, keep tables whole)
# ---------------------------------------------------------------------------
def items_to_chunks(items, source_file):
    chunks = []
    buffer = ""           # accumulates consecutive text under one section
    buf_page = None
    buf_section = ""
    idx = 0

    def flush():
        nonlocal buffer, buf_page, buf_section, idx
        if buffer.strip():
            c = build_chunk(buffer, source_file, buf_page, buf_section, "text", idx)
            if c:
                chunks.append(c)
                idx += 1
        buffer = ""

    for text, page, section, chunk_type in items:
        if chunk_type == "table":
            flush()
            c = build_chunk(text, source_file, page, section, "table", idx)
            if c:
                chunks.append(c)
                idx += 1
            continue

        # start a fresh buffer when the section changes
        if section != buf_section:
            flush()
            buf_section = section
            buf_page = page

        if buf_page is None:
            buf_page = page

        buffer += ("\n" if buffer else "") + text

        # split long buffers so chunks stay a reasonable size
        if len(buffer) >= config.CHUNK_CHAR_LIMIT:
            flush()
            buf_section = section
            buf_page = page

    flush()
    return chunks


def dedupe(chunks):
    seen, out = set(), []
    for c in chunks:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pdfs = sorted(config.DATA_RAW.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {config.DATA_RAW}. Add some and re-run.")
        return

    all_chunks = []
    for pdf_path in pdfs:
        source_file = pdf_path.name
        print(f"\nParsing {source_file} ...")

        try:
            items = parse_with_docling(pdf_path)
            print(f"  Docling OK ({len(items)} items)")
        except Exception as e:
            print(f"  Docling unavailable ({e}); using pdfplumber/PyMuPDF fallback")
            items = parse_with_fallback(pdf_path)
            print(f"  Fallback OK ({len(items)} items)")

        chunks = dedupe(items_to_chunks(items, source_file))
        print(f"  -> {len(chunks)} chunks")

        out_dir = config.DATA_PROCESSED / pdf_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "chunks.json").write_text(json.dumps(chunks, indent=2))
        all_chunks.extend(chunks)

    merged = config.DATA_PROCESSED / "merged_chunks.json"
    merged.write_text(json.dumps(all_chunks, indent=2))
    print(f"\nDone. {len(all_chunks)} chunks total -> {merged}")


if __name__ == "__main__":
    main()
