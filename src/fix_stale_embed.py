import chromadb

client = chromadb.PersistentClient(path="chroma_db")
name = "Ramesh Singh - Indian Economy - for Civil Services, Universities and Other Examinations-McGraw Hill Education (2018).pdf"

for strategy in ["section", "fixed"]:
    col = client.get_collection(f"chunks_{strategy}")
    col.delete(where={"source": name})
    print(f"{strategy}: deleted Ramesh Singh entries, now holds {col.count()}")