import json

from llm_gateway import call_llm

from retriever import hybrid_search, fuse_ranked_lists

REWRITE_PROMPT = """You are a search query optimizer for a retrieval system over Indian economics \
documents (RBI reports, Economic Surveys, NCERT textbooks, budget documents).

Rewrite the user's question into a single, clear, keyword-rich search query that will retrieve \
the most relevant passages. Keep it factual and specific. Do not answer the question — only \
rewrite it. Output ONLY the rewritten query, nothing else.

User question: {question}
Rewritten query:"""

MULTI_QUERY_PROMPT = """You are helping a retrieval system over Indian economics documents \
(RBI reports, Economic Surveys, NCERT textbooks, budget documents).

Generate {n} different search queries that approach the user's question from different angles \
(different phrasing, related sub-questions, different keywords) to maximize the chance of \
retrieving relevant passages. Keep each query short and keyword-focused.

Output ONLY a JSON array of {n} strings, nothing else. Example format: ["query one", "query two"]

User question: {question}
JSON array:"""


def rewrite_query(question):
    text, model_used = call_llm(
        [{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
        temperature=0.0, max_tokens=300, reasoning_effort="low",
    )
    return text.strip().strip('"'), model_used


def generate_multi_queries(question, n=3):
    text, model_used = call_llm(
        [{"role": "user", "content": MULTI_QUERY_PROMPT.format(question=question, n=n)}],
        temperature=0.3, max_tokens=400, reasoning_effort="low",
    )
    try:
        queries = json.loads(text.strip())
        if not isinstance(queries, list):
            raise ValueError("not a list")
    except (json.JSONDecodeError, ValueError):
        queries = [question]  # fallback: just use the original question
    return queries, model_used

def enhanced_search(question, strategy="section", final_n=5, per_query_k=10, n_multi=3):
    """
    Full query-enhanced retrieval: expands the raw question into several
    search queries (the original question + a clean rewrite + n_multi
    diverse multi-query variants), runs hybrid search independently for
    each, then fuses the ranked lists with RRF so chunks confirmed by
    multiple phrasings outrank chunks found by only one.

    Returns (fused_results, queries_used).
    """
    rewritten, _ = rewrite_query(question)
    multi_queries, _ = generate_multi_queries(question, n=n_multi)

    all_queries = [question, rewritten] + multi_queries
    seen, queries = set(), []
    for q in all_queries:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            queries.append(q)

    ranked_lists = [hybrid_search(q, strategy=strategy, final_n=per_query_k) for q in queries]
    return fuse_ranked_lists(ranked_lists, top_n=final_n), queries


if __name__ == "__main__":
    q = "what's up with food prices lately?"

    rewritten, model1 = rewrite_query(q)
    print(f"Original:  {q}")
    print(f"Rewritten: {rewritten}  [{model1}]")

    multi, model2 = generate_multi_queries(q, n=3)
    print(f"\nMulti-query variants [{model2}]:")
    for i, mq in enumerate(multi, 1):
        print(f"  {i}. {mq}")

    results, queries_used = enhanced_search(q)
    print(f"\nQueries used for fused retrieval ({len(queries_used)}):")
    for i, uq in enumerate(queries_used, 1):
        print(f"  {i}. {uq}")
    print(f"\nTop {len(results)} fused results:")
    for r in results:
        print(f"  {r['document']} p.{r['page']}  ({r['strategy']})")