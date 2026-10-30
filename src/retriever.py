from functools import lru_cache

import chromadb
from fastembed.rerank.cross_encoder import TextCrossEncoder

from ingest import CHROMA_DIR, embed_texts, load_bm25, tokenize

RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RRF_K = 60

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = TextCrossEncoder(model_name=RERANK_MODEL)
    return _reranker


@lru_cache(maxsize=1)
def get_client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


@lru_cache(maxsize=2)
def get_collection(strategy):
    return get_client().get_collection(f"chunks_{strategy}")


@lru_cache(maxsize=2)
def get_bm25(strategy):
    return load_bm25(strategy)


def dense_search(query, strategy="section", k=15, where=None):
    col = get_collection(strategy)
    q_emb = embed_texts([query])
    res = col.query(query_embeddings=q_emb, n_results=k, where=where)
    return [
        {"chunk_id": i, "text": d, **m}
        for i, d, m in zip(res["ids"][0], res["documents"][0], res["metadatas"][0])
    ]


def bm25_search(query, strategy="section", k=15, where=None):
    bm25, chunks = get_bm25(strategy)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in ranked:
        c = chunks[i]
        if where and any(c.get(k2) != v for k2, v in where.items()):
            continue
        out.append(c)
        if len(out) >= k:
            break
    return out


def rrf_fuse(dense_list, sparse_list, dense_weight=0.5, sparse_weight=0.5, top_n=20, stale_penalty=0.6):
    scores, by_id = {}, {}
    for rank, c in enumerate(dense_list):
        scores[c["chunk_id"]] = scores.get(c["chunk_id"], 0) + dense_weight / (RRF_K + rank + 1)
        by_id[c["chunk_id"]] = c
    for rank, c in enumerate(sparse_list):
        scores[c["chunk_id"]] = scores.get(c["chunk_id"], 0) + sparse_weight / (RRF_K + rank + 1)
        by_id[c["chunk_id"]] = c

    for cid, c in by_id.items():
        if c.get("stale_risk"):
            scores[cid] *= stale_penalty

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return [by_id[i] for i in ranked_ids]

def rerank(query, candidates, top_n=5, stale_penalty=0.6):
    if not candidates:
        return []
    scores = list(get_reranker().rerank(query, [c["text"] for c in candidates]))
    scores = [s * stale_penalty if c.get("stale_risk") else s for s, c in zip(scores, candidates)]
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [candidates[i] for i in order]


def hybrid_search(query, strategy="section", dense_k=15, sparse_k=15, fuse_n=20,
                   final_n=5, dense_weight=0.5, sparse_weight=0.5,
                   use_rerank=True, where=None):
    dense = dense_search(query, strategy, k=dense_k, where=where)
    sparse = bm25_search(query, strategy, k=sparse_k, where=where)
    fused = rrf_fuse(dense, sparse, dense_weight, sparse_weight, top_n=fuse_n)
    return rerank(query, fused, top_n=final_n) if use_rerank else fused[:final_n]

def fuse_ranked_lists(ranked_lists, top_n=10):
    """
    Fuses multiple ranked lists of chunks (e.g. hybrid_search results from
    different query-variant phrasings) using Reciprocal Rank Fusion, so a
    chunk retrieved near the top by several query variants ranks higher
    than one retrieved by only a single phrasing.
    """
    scores, by_id = {}, {}
    for ranked in ranked_lists:
        for rank, c in enumerate(ranked):
            scores[c["chunk_id"]] = scores.get(c["chunk_id"], 0) + 1 / (RRF_K + rank + 1)
            by_id[c["chunk_id"]] = c
    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return [by_id[i] for i in ranked_ids]


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "repo rate"
    for strategy in ("section", "fixed"):
        print(f"\n=== {strategy} ===")
        for c in hybrid_search(q, strategy=strategy, final_n=5):
            print(f"[{c['document']} ({c['year']})] p.{c['page']}  stale_risk={c.get('stale_risk')}")
            print(f"   {c['text'][:150].replace(chr(10), ' ')}")