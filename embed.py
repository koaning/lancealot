import lancedb
from lancedb.pydantic import Vector, LanceModel
import modal
import srsly
from itertools import islice
from sentence_transformers import SentenceTransformer

n_arxiv = 67140


def batched(iterable, n=10):
    "Batch data into lists of length n. The last batch may be shorter."
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


stub = modal.Stub("example-get-started")
image = (modal.Image.debian_slim()
         .pip_install("sentence_transformers", "srsly", "lancedb", "polars", "pysbd")
         .run_commands("python -c 'from sentence_transformers import SentenceTransformer; tfm = SentenceTransformer(\"all-MiniLM-L6-v2\")'"))

def add_vectors(batches, tfm, col):
        for batch in batches:
            vectors = fetch_vectors.remote(batch, tfm, col)
            yield [{"vector": vec, **ex} for ex, vec in zip(batch, vectors)]

class Sentence(LanceModel):
    vector: Vector(368)
    text: str
    meta: dict
    
@stub.function(image=image, gpu="any")
def fetch_vectors(batch, tfm, col):
    tfm = SentenceTransformer(tfm)
    return tfm.encode([ex[col] for ex in batch])


@stub.local_entrypoint()
def main():
    db = lancedb.connect("./.lancedb")

    model = "tomaarsen/mpnet-base-nli-matryoshka"
    batch_size = 10_000

    batches = batched(srsly.load_jsonl("sentences.jsonl"), n=batch_size)
    with_vecs = add_vectors(batches, model, col="text")

    batch = next(with_vecs)
    tbl = db.create_table(
        "arxiv-sentences", 
        exist_ok=True, 
        schema=Sentence, 
        data=batch
    )
    for batch in with_vecs:
        tbl.add(batch)
