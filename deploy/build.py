#!/usr/bin/env python3
"""
build.py — Build a txtai embeddings index with graph RAG support.

Run this on the indexing machine.  The output is a single portable .tar.gz
archive that can be copied to the serving machine and loaded by txtai at
startup without any additional configuration.

Usage
-----
    # Index a directory of plain-text, Markdown, and/or PDF files:
    python build.py --source /path/to/docs --output index.tar.gz

    # Index PDFs with smaller chunks (default is 10 sentences per chunk):
    python build.py --source /path/to/pdfs --chunk-sentences 5 --output index.tar.gz

    # Index a CSV file (columns: id, text, [tags]):
    python build.py --source corpus.csv --output index.tar.gz

    # Incremental upsert into an existing archive:
    python build.py --source new-docs/ --index existing.tar.gz --output updated.tar.gz

Serving
-------
    Copy index.tar.gz to the server, set `path: /data/index.tar.gz` in
    config.yml, and start docker compose.  txtai auto-extracts the archive.

Requirements
------------
    pip install "txtai[api,graph,vectors,pipeline-data]"

    For PDF extraction (install one):
        pip install pymupdf          # recommended — faster, better quality
        pip install pdfminer.six     # fallback

    The embedding model (sentence-transformers/all-MiniLM-L6-v2) is
    downloaded automatically on first run and cached under ~/.cache/torch.
    Use HF_HOME to override the cache location.

    The model name must match the one in config.yml on the serving machine.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import nltk

nltk.download("punkt_tab", quiet=True)

from txtai import Embeddings
from txtai.pipeline import Textractor, Segmentation


# ── Embedding model ───────────────────────────────────────────────────────────
# Must match config.yml on the serving machine.
MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"

# ── Index configuration ───────────────────────────────────────────────────────
INDEX_CONFIG = {
    "path": MODEL_PATH,
    "content": True,
    "scoring": {
        "method": "bm25",
        "terms": True,
    },
    "graph": {
        "minscore": 0.15,
        "topics": {
            "algorithm": "louvain",
            "terms": 4,
        },
    },
}


def load_text_files(source: Path):
    """
    Yield (id, text, None) tuples from every .txt / .md file under `source`.

    The document ID is the relative file path so that subsequent upserts
    can overwrite individual documents by path.
    """
    extensions = {".txt", ".md", ".rst"}
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                yield str(path.relative_to(source)), text, None


def load_pdf_files(source: Path, chunk_sentences: int = 10):
    """
    Yield (id, {text, filename}, None) tuples from every .pdf file under `source`.

    Each PDF is split into chunks of `chunk_sentences` sentences so that long
    documents produce focused, retrievable passages rather than one giant blob.
    The document ID encodes the filename and chunk index for stable upserts.
    """
    textractor = Textractor(backend=None)
    segmentation = Segmentation(sentences=True)

    for path in sorted(source.rglob("*.pdf")):
        rel = str(path.relative_to(source))
        print(f"  Extracting {rel} …")
        try:
            text = textractor(str(path))
        except Exception as exc:
            print(f"  WARNING: could not extract {rel}: {exc}", file=sys.stderr)
            continue

        sentences = segmentation(text)
        if isinstance(sentences, str):
            sentences = [sentences]

        # Group sentences into chunks of chunk_sentences
        sentence_groups = [sentences[i : i + chunk_sentences] for i in range(0, len(sentences), chunk_sentences)]

        for i, group in enumerate(sentence_groups):
            chunk = " ".join(s for s in group if isinstance(s, str)).strip()
            if chunk:
                doc_id = f"{rel}::chunk{i}"
                yield doc_id, {"text": chunk, "filename": rel}, None


def load_csv_file(source: Path):
    """
    Yield (id, text, tags) tuples from a CSV file.

    Expected columns (header row required):
        id    — unique document identifier
        text  — document body
        tags  — optional; comma-separated tag string
    """
    with open(source, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "text" not in (reader.fieldnames or []):
            raise ValueError(f"CSV must have a 'text' column; got: {reader.fieldnames}")
        for i, row in enumerate(reader):
            doc_id = row.get("id") or str(i)
            text = row.get("text", "").strip()
            tags = row.get("tags", None)
            if text:
                yield doc_id, text, tags


def iter_documents(source: str, chunk_sentences: int = 10):
    """Route to the appropriate loader based on the source path."""
    path = Path(source)
    if not path.exists():
        print(f"ERROR: source path does not exist: {path}", file=sys.stderr)
        sys.exit(1)

    if path.is_dir():
        yield from load_text_files(path)
        yield from load_pdf_files(path, chunk_sentences=chunk_sentences)
    elif path.suffix.lower() == ".csv":
        yield from load_csv_file(path)
    elif path.suffix.lower() == ".pdf":
        yield from load_pdf_files(path.parent, chunk_sentences=chunk_sentences)
    else:
        print(
            f"ERROR: unsupported source type '{path.suffix}'. "
            "Provide a directory of .txt/.md/.pdf files or a .csv file.",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Build a txtai index archive.")
    parser.add_argument("--source", required=True, help="Source directory or CSV file to index.")
    parser.add_argument("--output", default="index.tar.gz", help="Output archive path (default: index.tar.gz).")
    parser.add_argument(
        "--index",
        default=None,
        help="Existing index archive to load before upserting (incremental build).",
    )
    parser.add_argument(
        "--chunk-sentences",
        type=int,
        default=10,
        help="Sentences per chunk when splitting PDFs (default: 10).",
    )
    args = parser.parse_args()

    # ── Load or create embeddings ─────────────────────────────────────────────
    if args.index:
        print(f"Loading existing index from {args.index} …")
        embeddings = Embeddings()
        embeddings.load(args.index)
    else:
        print(f"Creating new index with model '{MODEL_PATH}' …")
        embeddings = Embeddings(INDEX_CONFIG)

    # ── Collect documents ─────────────────────────────────────────────────────
    docs = list(iter_documents(args.source, chunk_sentences=args.chunk_sentences))
    if not docs:
        print("WARNING: no documents found in source. Nothing to index.", file=sys.stderr)
        sys.exit(0)

    print(f"Indexing {len(docs)} document(s) …")

    # ── Index or upsert ───────────────────────────────────────────────────────
    if args.index:
        embeddings.upsert(docs)
    else:
        embeddings.index(docs)

    # ── Save archive ──────────────────────────────────────────────────────────
    print(f"Saving index to {args.output} …")
    embeddings.save(args.output)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Done. Archive size: {size_mb:.1f} MB")
    print()
    print("Next steps:")
    print(f"  scp {args.output} user@server:/data/")
    print("  Update config.yml: path: /data/index.tar.gz")
    print("  docker compose up -d")


if __name__ == "__main__":
    main()
