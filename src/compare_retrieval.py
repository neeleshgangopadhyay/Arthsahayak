import sys

from retriever import bm25_search, dense_search, hybrid_search, rerank, rrf_fuse


def show(label, results):
    print(f"\n-- {label} --")
    for c in results:
        print(f"[{c['document']} ({c['year']})] p.{c['page']}  stale_risk={c.get('stale_risk')}")
        print(f"   {c['text'][:150].replace(chr(10), ' ')}")


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "repo rate"
    strategy = sys.argv[2] if len(sys.argv) > 2 else "section"
    k = 5

    print(f"Query: {query!r}  strategy: {strategy}")

    dense_only = dense_search(query, strategy=strategy, k=k)
    show("1. Dense only", dense_only)

    dense = dense_search(query, strategy=strategy, k=15)
    sparse = bm25_search(query, strategy=strategy, k=15)
    fused = rrf_fuse(dense, sparse, top_n=20)
    show("2. Hybrid (dense + BM25 + RRF), no rerank", fused[:k])

    reranked = rerank(query, fused, top_n=k)
    show("3. Hybrid + cross-encoder rerank (final)", reranked)

    full = hybrid_search(query, strategy=strategy, final_n=k)
    show("4. hybrid_search() end-to-end (sanity check, should match #3)", full)


if __name__ == "__main__":
    main()