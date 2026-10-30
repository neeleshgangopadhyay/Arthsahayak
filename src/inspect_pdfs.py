import hashlib
from pathlib import Path
from pypdf import PdfReader

rows, seen = [], {}
for pdf in sorted(Path("data").glob("*.pdf")):
    try:
        reader = PdfReader(str(pdf))
        n = len(reader.pages)
        step = max(1, n // 30)  # sample up to ~30 pages
        texts = [(reader.pages[i].extract_text() or "") for i in range(0, n, step)]
        empty = sum(1 for t in texts if len(t.strip()) < 50) / len(texts)
        chars = sum(len(t) for t in texts) // len(texts)
        head = "".join((reader.pages[i].extract_text() or "") for i in range(min(3, n)))
        digest = hashlib.md5(head.strip().encode()).hexdigest()
        flag = "SCANNED?" if empty > 0.5 else ""
        if head.strip() and digest in seen:
            flag += f" DUPLICATE of {seen[digest]}"
        seen[digest] = pdf.name
        rows.append((pdf.name, n, chars, f"{empty:.0%}", flag))
    except Exception as e:
        rows.append((pdf.name, "?", "?", "?", f"ERROR: {str(e)[:50]}"))

print(f"{'file':45} {'pages':>6} {'chars/pg':>9} {'empty':>6}  flags")
for r in rows:
    print(f"{r[0][:45]:45} {r[1]:>6} {r[2]:>9} {r[3]:>6}  {r[4]}")
print("\nTotal pages:", sum(r[1] for r in rows if isinstance(r[1], int)))