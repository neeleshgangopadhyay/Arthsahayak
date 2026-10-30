import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from retriever import bm25_search, dense_search, hybrid_search, rerank, rrf_fuse

QUESTIONS_FILE = Path("eval/questions.json")
RESULTS_DIR = Path("eval/results")


def load_questions():
    return json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))


def retrieve(query, strategy, mode, k=5):
    if mode == "dense":
        return dense_search(query, strategy=strategy, k=k)
    dense = dense_search(query, strategy=strategy, k=15)
    sparse = bm25_search(query, strategy=strategy, k=15)
    fused = rrf_fuse(dense, sparse, top_n=20)
    if mode == "hybrid":
        return fused[:k]
    return rerank(query, fused, top_n=k)  # mode == "hybrid_rerank"


def is_hit(results, q):
    pages = q["page"] if isinstance(q["page"], list) else [q["page"]]
    return any(c["document"] == q["document"] and c["page"] in pages for c in results)

def run_eval(strategy, mode):
    questions = load_questions()
    per_type = defaultdict(lambda: [0, 0])  # type -> [hits, total]
    details = []

    for q in questions:
        if q["type"] == "unanswerable":
            continue  # hit-rate doesn't apply; handled separately later (agent abstention test)
        results = retrieve(q["question"], strategy, mode)
        hit = is_hit(results, q)
        per_type[q["type"]][0] += int(hit)
        per_type[q["type"]][1] += 1
        details.append({
            "id": q["id"], "type": q["type"], "question": q["question"],
            "expected": f"{q['document']} p.{q['page']}", "hit": hit,
            "retrieved": [f"{c['document']} p.{c['page']}" for c in results],
        })

    total_hits = sum(h for h, _ in per_type.values())
    total_n = sum(n for _, n in per_type.values())
    overall = total_hits / total_n if total_n else 0.0

    return {
        "strategy": strategy, "mode": mode, "overall_hit_rate@5": round(overall, 3),
        "by_type": {t: round(h / n, 3) for t, (h, n) in per_type.items()},
        "counts_by_type": {t: f"{h}/{n}" for t, (h, n) in per_type.items()},
        "details": details,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["fixed", "section"], default="section")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for mode in ("dense", "hybrid", "hybrid_rerank"):
        result = run_eval(args.strategy, mode)
        all_results[mode] = result
        print(f"\n=== strategy={args.strategy}  mode={mode} ===")
        print(f"Overall hit-rate@5: {result['overall_hit_rate@5']}")
        for t, rate in result["by_type"].items():
            print(f"  {t:15s} {rate}  ({result['counts_by_type'][t]})")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"retrieval_eval_{args.strategy}_{ts}.json"
    out_file.write_text(json.dumps(all_results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved full results (including per-question detail) to {out_file}")


if __name__ == "__main__":
    main()