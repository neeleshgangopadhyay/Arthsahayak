import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import chromadb
import fitz  # pymupdf
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding

import logging
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("fitz").setLevel(logging.ERROR)

DATA_DIR = Path("data")
INDEX_DIR = Path("index")
CACHE_DIR = INDEX_DIR / "cache"
CHROMA_DIR = "chroma_db"
META_FILE = DATA_DIR / "sources_meta.json"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
PIPELINE_VERSION = "3"  # bump this when you change cleaning/chunking/embedding code


# ---------- Embedding (torch-free, via fastembed/ONNX) ----------
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
    return _embedder


def embed_texts(texts):
    return [v.tolist() for v in get_embedder().embed(texts)]


def load_doc_info():
    if META_FILE.exists():
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    return {}


# ---------- Load / clean ----------
def _pypdf_pages(path):
    return [p.extract_text() or "" for p in PdfReader(str(path)).pages]


def _pymupdf_pages(path):
    doc = fitz.open(str(path))
    return [page.get_text("text") for page in doc]


def _broken_word_rate(sample_text):
    """Fraction of words that look like 'mid dle' or 'y ou' — split by a lone space."""
    words = sample_text.split()
    if len(words) < 20:
        return 0.0
    suspicious = sum(
        1 for i in range(len(words) - 1)
        if len(words[i]) <= 3 and words[i].isalpha()
        and len(words[i + 1]) <= 3 and words[i + 1].isalpha()
        and not words[i][0].isupper()
    )
    return suspicious / len(words)


def load_pdf(path, info):
    pypdf_pages = _pypdf_pages(path)
    sample = "\n".join(pypdf_pages[:3])
    use_pymupdf = _broken_word_rate(sample) > 0.03  # tune if needed

    if use_pymupdf:
        pages_text = _pymupdf_pages(path)
        print(f"  [{path.name}] using pymupdf (pypdf looked broken)")
    else:
        pages_text = pypdf_pages

    return [
        {"text": text, "document": info["document"], "year": info["year"],
         "page": i + 1, "source": path.name, "stale_risk": info.get("stale_risk", False)}
        for i, text in enumerate(pages_text)
    ]


def find_repeated_lines(pages, min_share=0.3):
    counts = Counter()
    for p in pages:
        counts.update({l.strip() for l in p["text"].splitlines() if l.strip()})
    threshold = max(3, int(len(pages) * min_share))
    return {l for l, c in counts.items() if c >= threshold and len(l) < 100}


def reflow(text):
    paras, buf = [], []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            if buf:
                paras.append(" ".join(buf)); buf = []
            continue
        buf.append(s)
        if re.search(r"[.!?:]$", s) or len(s) < 60:
            paras.append(" ".join(buf)); buf = []
    if buf:
        paras.append(" ".join(buf))
    return "\n\n".join(paras)


def clean_page(text, repeated):
    lines = [l.strip() for l in text.splitlines()
             if l.strip() not in repeated and not re.fullmatch(r"\d{1,4}", l.strip())]
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = reflow(text)
    return re.sub(r"[ \t]+", " ", text).strip()


# ---------- Chunk ----------
def make_splitter(strategy):
    if strategy == "fixed":
        return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120, separators=[" ", ""])
    return RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""])


def process_pdf(path, info, strategy):
    pages = load_pdf(path, info)
    repeated = find_repeated_lines(pages)
    splitter = make_splitter(strategy)
    chunks = []
    for p in pages:
        text = clean_page(p["text"], repeated)
        if len(text) < 50:
            continue
        for i, piece in enumerate(splitter.split_text(text)):
            if len(piece.strip()) < 80:
                continue
            chunks.append({
                "chunk_id": f"{strategy}-{path.stem}-p{p['page']}-{i}",
                "text": piece, "document": p["document"], "year": p["year"],
                "page": p["page"], "source": p["source"], "strategy": strategy,
                "stale_risk": p["stale_risk"],
            })
    return chunks


# ---------- Index helpers (also used by retriever / check_index) ----------
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def load_chunks(strategy):
    with open(INDEX_DIR / f"chunks_{strategy}.json", encoding="utf-8") as f:
        return json.load(f)


def load_bm25(strategy):
    chunks = load_chunks(strategy)
    return BM25Okapi([tokenize(c["text"]) for c in chunks]), chunks


def signature(path, info):
    raw = f"{PIPELINE_VERSION}|{path.stat().st_size}|{int(path.stat().st_mtime)}|{json.dumps(info, sort_keys=True)}"
    return hashlib.sha1(raw.encode()).hexdigest()


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["fixed", "section"], default="section")
    ap.add_argument("--rebuild", action="store_true", help="ignore the cache and reprocess every PDF")
    args = ap.parse_args()
    strategy = args.strategy

    (CACHE_DIR / strategy).mkdir(parents=True, exist_ok=True)
    infos = load_doc_info()
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    print(f"{len(pdfs)} PDFs found. Strategy: {strategy}\n")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(f"chunks_{strategy}", metadata={"hnsw:space": "cosine"})
    all_chunks, failed = [], []

    for n, pdf in enumerate(pdfs, 1):
        t0 = time.time()
        info = infos.get(pdf.name) or {"document": pdf.stem, "year": "unknown"}
        sig = signature(pdf, info)
        cache_file = CACHE_DIR / strategy / f"{pdf.stem}.json"

        chunks, status, freshly_processed = None, "cached", False
        if cache_file.exists() and not args.rebuild:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached["signature"] == sig:
                chunks = cached["chunks"]
        if chunks is None:
            status, freshly_processed = "processed", True
            try:
                chunks = process_pdf(pdf, info, strategy)
            except Exception as e:
                failed.append((pdf.name, f"error: {str(e)[:80]}"))
                print(f"[{n}/{len(pdfs)}] {pdf.name}: FAILED ({e})")
                continue
            if not chunks:
                failed.append((pdf.name, "no text extracted (scanned?)"))
                print(f"[{n}/{len(pdfs)}] {pdf.name}: no text, skipped")
                continue
            cache_file.write_text(json.dumps({"signature": sig, "chunks": chunks}, ensure_ascii=False), encoding="utf-8")

        # re-embed whenever freshly reprocessed (content/metadata may have changed even
        # if the chunk count didn't), or whenever the stored count doesn't match
        present = len(col.get(where={"source": pdf.name}, include=[])["ids"])
        if freshly_processed or present != len(chunks):
            status = "processed+embedded" if freshly_processed else "embedded"
            if present:
                col.delete(where={"source": pdf.name})
            for i in range(0, len(chunks), 500):
                batch = chunks[i:i + 500]
                col.add(
                    ids=[c["chunk_id"] for c in batch],
                    documents=[c["text"] for c in batch],
                    embeddings=embed_texts([c["text"] for c in batch]),
                    metadatas=[{k: c[k] for k in ("document", "year", "page", "source", "strategy", "stale_risk")} for c in batch],
                )
        all_chunks.extend(chunks)
        print(f"[{n}/{len(pdfs)}] {pdf.name}: {len(chunks)} chunks ({status}, {time.time() - t0:.0f}s)")

    # remove chunks of PDFs that were deleted from data/
    stored = {m["source"] for m in col.get(include=["metadatas"])["metadatas"]}
    for gone in stored - {p.name for p in pdfs}:
        col.delete(where={"source": gone})
        print(f"Removed deleted PDF from index: {gone}")

    INDEX_DIR.mkdir(exist_ok=True)
    (INDEX_DIR / f"chunks_{strategy}.json").write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    (INDEX_DIR / f"ingest_report_{strategy}.json").write_text(
        json.dumps({"total_chunks": len(all_chunks), "failed": failed}, indent=1), encoding="utf-8")

    print(f"\nDone: {len(all_chunks)} chunks from {len(pdfs) - len(failed)} PDFs; chroma holds {col.count()}.")
    if failed:
        print("FAILED / SKIPPED:")
        for name, why in failed:
            print(f"  - {name}: {why}")


if __name__ == "__main__":
    main()