"""
Ingestion + embedding stage of the RAG pipeline.

    python ingest.py          # (re)build the ChromaDB index from docs/*.txt

- load:   every docs/doc_XX.txt file
- chunk:  one chunk per document (each policy is a single ~80-word paragraph, so splitting it would
          only separate facts that belong together, e.g. a fee from the condition it applies to)
- embed:  sentence-transformers all-MiniLM-L6-v2 (local, no API key)
- store:  ChromaDB persistent collection "zepto_policies" using cosine distance
"""

import os
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "docs"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", HERE / "chroma_db"))
COLLECTION_NAME = "zepto_policies"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DOC_TITLES = {
    "doc_01": "Delivery Policy",
    "doc_02": "Returns & Refunds",
    "doc_03": "Membership Tiers",
    "doc_04": "Order Tracking",
    "doc_05": "Order Cancellation Policy",
    "doc_06": "Damaged or Missing Items",
    "doc_07": "Gift Cards",
    "doc_08": "Customer Support Hours",
}


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME, device="cpu")


def embed(texts):
    # normalize -> unit vectors, so cosine similarity == dot product
    return get_embedder().encode(list(texts), normalize_embeddings=True).tolist()


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def load_and_chunk():
    """Return (ids, texts, metadatas) - one chunk per document."""
    ids, texts, metas = [], [], []
    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        doc_id = path.stem
        text = path.read_text(encoding="utf-8").strip()
        ids.append(f"{doc_id}_chunk_0")
        texts.append(text)
        metas.append({"doc_id": doc_id, "title": DOC_TITLES.get(doc_id, doc_id), "source": path.name})
    return ids, texts, metas


def build_index():
    ids, texts, metas = load_and_chunk()
    collection = get_collection()
    collection.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=embed(texts))
    return collection.count()


def ensure_index():
    """Build the index on first use (e.g. a fresh clone or container) - idempotent."""
    if get_collection().count() < len(list(DOCS_DIR.glob("doc_*.txt"))):
        build_index()


def retrieve(query: str, k: int = 3):
    """Embed the query and return the top-k chunks by cosine similarity (always real, in both modes)."""
    ensure_index()
    res = get_collection().query(query_embeddings=embed([query]), n_results=k,
                                 include=["documents", "metadatas", "distances"])
    return [
        {"id": cid, "doc_id": meta["doc_id"], "title": meta["title"], "text": doc,
         "similarity": round(1 - dist, 4)}           # cosine distance -> cosine similarity
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


if __name__ == "__main__":
    n = build_index()
    print(f"Indexed {n} chunks into ChromaDB collection '{COLLECTION_NAME}' at {CHROMA_DIR}")
    for q in ["How much is the delivery fee?", "Can I get cash for my gift card?"]:
        print(f"\nQuery: {q}")
        for hit in retrieve(q):
            print(f"  {hit['id']:<16} sim={hit['similarity']:.3f}  [{hit['title']}]")
