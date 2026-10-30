import random
import sys

from ingest import load_chunks

strategy = sys.argv[1] if len(sys.argv) > 1 else "section"
doc_filter = sys.argv[2] if len(sys.argv) > 2 else None
n = int(sys.argv[3]) if len(sys.argv) > 3 else 8

chunks = load_chunks(strategy)
if doc_filter:
    chunks = [c for c in chunks if doc_filter.lower() in c["document"].lower()]

sample = random.sample(chunks, min(n, len(chunks)))
for c in sample:
    print(f"\n[{c['document']} ({c['year']})]  page {c['page']}")
    print(f"  {c['text'][:300].replace(chr(10), ' ')}")