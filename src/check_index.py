import argparse
from collections import Counter

import chromadb

from ingest import CHROMA_DIR, embed_texts, load_bm25, tokenize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default="repo rate", help="test query")
    ap.add_argument("--strategy", choices=["fixed", "section", "both"], default="both")
    ap.add_argument("--k", type=int, default=3, help="results to show per method")
    args = ap.parse_args()

    strategies = ["fixed", "section"] if args.strategy == "both" else [args.strategy]
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    for strategy in strategies:
        print(f"\n{'=' * 70}\nSTRATEGY: {strategy}\n{'=' * 70}")
        try:
            bm25, chunks = load_bm25(strategy)
            col = client.get_collection(f"chunks_{strategy}")
        except Exception as e:
            print(f"Cannot load '{strategy}' index: {e}")
            print(f"Run: python src/ingest.py --strategy {strategy}")
            continue

        # 1. The three stores must agree
        n_json, n_chroma, n_bm25 = len(chunks), col.count(), bm25.corpus_size
        status = "OK" if n_json == n_chroma == n_bm25 else "MISMATCH, re-run ingest with --rebuild"
        print(f"chunks.json: {n_json} | chroma: {n_chroma} | bm25: {n_bm25}  -> {status}")

        # 2. Chunk size and spread over documents
        lengths = [len(c["text"]) for c in chunks]
        print(f"chunk length (chars): min {min(lengths)}, "
              f"avg {sum(lengths) // len(lengths)}, max {max(lengths)}")
        per_doc = Counter(f"{c['document']} ({c['year']})" for c in chunks)
        print(f"documents indexed: {len(per_doc)}")
        for name, n in per_doc.most_common(5):
            print(f"   {n:>6} chunks  {name}")
        if len(per_doc) > 5:
            print(f"   ... and {len(per_doc) - 5} more")

        # 3. BM25 results
        print(f"\nQuery: {args.query!r}")
        scores = bm25.get_scores(tokenize(args.query))
        top = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:args.k]
        print(f"\n-- BM25 top {args.k} --")
        for i in top:
            c = chunks[i]
            print(f"[{scores[i]:.2f}] {c['document']} ({c['year']}) p.{c['page']}")
            print(f"        {c['text'][:200].replace(chr(10), ' ')}")

        # 4. Dense results
        q = embed_texts([args.query])
        res = col.query(query_embeddings=q, n_results=args.k)
        print(f"\n-- Dense top {args.k} --")
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            print(f"[{1 - dist:.2f}] {meta['document']} ({meta['year']}) p.{meta['page']}")
            print(f"        {doc[:200].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()