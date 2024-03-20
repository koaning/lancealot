import lancedb 
from sentence_transformers import SentenceTransformer

db = lancedb.connect(".lancedb")
tbl = db.open_table("arxiv-sentences")
tfm = SentenceTransformer("all-MiniLM-L6-v2")

tbl.query(tfm.encode(["new dataset"])[0]).limit(10)
print([_['text'] for _ in tbl.search(tfm.encode(["new dataset"])[0]).limit(100).to_list()])