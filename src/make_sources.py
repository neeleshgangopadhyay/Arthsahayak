import json
import re
from datetime import date
from pathlib import Path

from pypdf import PdfReader

DATA_DIR = Path("data")
META_FILE = DATA_DIR / "sources_meta.json"
OUT_FILE = DATA_DIR / "SOURCES.md"
TODO = "TODO"

# keyword found in first pages -> issuer name
ISSUERS = {
    "reserve bank of india": "Reserve Bank of India",
    "ministry of finance": "Ministry of Finance",
    "ministry of statistics": "MoSPI",
    "niti aayog": "NITI Aayog",
    "ncert": "NCERT",
    "finance commission": "Finance Commission",
    "department of economic affairs": "Department of Economic Affairs",
}
YEAR_RE = re.compile(r"(20\d{2}(?:\s?[-–/]\s?\d{2,4})?)")


def first_pages_text(reader, n=2):
    out = []
    for page in reader.pages[:n]:
        try:
            out.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(out)


def guess_fields(pdf):
    """Best-effort guesses. The user reviews them."""
    guess = {
        "document": pdf.stem.replace("_", " ").replace("-", " ").title(),
        "year": TODO, "issuer": TODO, "pages": "?",
    }
    try:
        reader = PdfReader(str(pdf))
        guess["pages"] = len(reader.pages)
        title = (reader.metadata.title or "").strip() if reader.metadata else ""
        if 8 <= len(title) <= 120:
            guess["document"] = title
        text = first_pages_text(reader)
    except Exception:
        text = ""

    m = YEAR_RE.search(pdf.stem.replace("_", "-")) or YEAR_RE.search(text)
    if m:
        guess["year"] = re.sub(r"\s", "", m.group(1)).replace("–", "-")

    low = text.lower()
    for key, name in ISSUERS.items():
        if key in low:
            guess["issuer"] = name
            break
    return guess


def load_meta():
    return json.loads(META_FILE.read_text(encoding="utf-8")) if META_FILE.exists() else {}


def main():
    DATA_DIR.mkdir(exist_ok=True)
    meta = load_meta()
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    found = {p.name for p in pdfs}

    for pdf in pdfs:
        if pdf.name not in meta:
            g = guess_fields(pdf)
            meta[pdf.name] = {
                "document": g["document"], "year": g["year"], "issuer": g["issuer"],
                "url": TODO, "edition": TODO,
                "downloaded": date.today().strftime("%d-%m-%Y"),
                "notes": "auto-guessed, please review",
            }
            print(f"NEW: {pdf.name} -> {g['document']} | {g['year']} | {g['issuer']}")

    for name in meta:
        if name not in found:
            print(f"WARNING: {name} is in sources_meta.json but not in data/")

    META_FILE.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Data Sources", "",
        "Official publications only. Evaluation questions are tied to these exact editions.", "",
        "| File | Document | Year | Issuing body | Source URL | Edition | Downloaded | Pages | MB |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    todo_count = 0
    for pdf in pdfs:
        m = meta[pdf.name]
        try:
            pages = len(PdfReader(str(pdf)).pages)
        except Exception:
            pages = "?"
        mb = f"{pdf.stat().st_size / 1_048_576:.1f}"
        todo_count += sum(1 for k in ("year", "issuer", "url", "edition") if m[k] == TODO)
        lines.append(
            f"| {pdf.name} | {m['document']} | {m['year']} | {m['issuer']} | {m['url']} "
            f"| {m['edition']} | {m['downloaded']} | {pages} | {mb} |"
        )
    OUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_FILE}: {len(pdfs)} documents, {todo_count} TODO fields left.")


if __name__ == "__main__":
    main()