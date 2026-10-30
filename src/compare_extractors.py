import sys
import fitz  # pymupdf
from pypdf import PdfReader

path, page_no = sys.argv[1], int(sys.argv[2])  # page_no is 1-based
a = PdfReader(path).pages[page_no - 1].extract_text() or ""
b = fitz.open(path)[page_no - 1].get_text("text")
print("=== pypdf ===\n", a[:800])
print("\n=== pymupdf ===\n", b[:800])