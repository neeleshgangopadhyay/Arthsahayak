import glob
import json

path = sorted(glob.glob("eval/results/retrieval_eval_section_*.json"))[-1]
data = json.load(open(path, encoding="utf-8"))

dense = {d["id"]: d for d in data["dense"]["details"]}
hybrid = {d["id"]: d for d in data["hybrid_rerank"]["details"]}

print(f"Comparing dense vs hybrid_rerank from: {path}\n")
for qid in dense:
    d, h = dense[qid], hybrid[qid]
    if d["hit"] != h["hit"]:
        print(f"{qid}: {d['question']}")
        print(f"  expected: {d['expected']}")
        print(f"  dense ({'HIT' if d['hit'] else 'MISS'}):         {d['retrieved']}")
        print(f"  hybrid_rerank ({'HIT' if h['hit'] else 'MISS'}): {h['retrieved']}")
        print()