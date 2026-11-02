"""
RAG knowledge base — Chroma vector store + embedding + retrieval.

Why RAG, bottom-up:
    1. Why RAG at all: CNC XML syntax, controls, and binding rules are private,
       frequently-updated knowledge that is absent from the LLM's training data
       and should not be stuffed into the prompt (it would blow up the context
       window and cannot be incrementally updated).
    2. Embedding: map text into a high-dimensional vector where semantic
       similarity corresponds to a small vector angle (high cosine similarity).
       BGE-small-zh is chosen for strong Chinese semantic retrieval, a small
       dimension (512) for fast search, and the ability to run locally.
    3. Retrieval: Chroma uses an HNSW approximate-nearest-neighbor index — log
       complexity — instead of brute-force scanning.
    4. Why Chroma over FAISS:
       - Chroma ships persistence + metadata filtering + incremental upsert,
         fitting a "knowledge changes" scenario;
       - FAISS excels at raw retrieval performance (large-scale, GPU) but is
         awkward for insert/delete and suits large static corpora.
       Classic trade-off: scale-first → FAISS, engineering-convenience-first → Chroma.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.core.config import settings

# Lazy imports: mock mode can import this module without chromadb / sentence-transformers.
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    _HAS_CHROMA = True
except ImportError:  # pragma: no cover
    _HAS_CHROMA = False

try:
    from sentence_transformers import SentenceTransformer

    _HAS_ST = True
except ImportError:  # pragma: no cover
    _HAS_ST = False


class _HashEmbedder:
    """Fallback embedder — character n-gram bag-of-words pseudo-vectors.

    Purpose: keep the vertical slice runnable without a GPU or
    sentence-transformers. Production swaps in BGE; this is just the fallback of
    a pluggable embedding interface. The fallback still preserves the core
    embedding property — "more text overlap → more similar vectors" — so mock
    mode can still demonstrate "semantic similarity → recall".
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    @staticmethod
    def _ngrams(text: str, n: int = 2):
        """Character n-gram feature set (Chinese by char, English by word)."""
        feats = set()
        for w in text.replace("，", " ").replace("。", " ").replace("；", " ").split():
            if w:
                feats.add(w)
                for i in range(len(w) - n + 1):
                    feats.add(w[i : i + n])
        # Fallback: also n-gram the whole text so Chinese chars always hit.
        for i in range(len(text) - n + 1):
            feats.add(text[i : i + n])
        return feats

    def encode(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for t in texts:
            v = [0.0] * self.dim
            for feat in self._ngrams(t):
                # Stable hash maps a feature onto a dimension, mimicking a sparse embedding.
                idx = hash(feat) % self.dim
                v[idx] += 1.0
            norm = (sum(x * x for x in v)) ** 0.5 or 1.0
            vecs.append([x / norm for x in v])
        return vecs


class VectorStore:
    """Chroma vector-store wrapper. Exposes index building and similarity search."""

    def __init__(self, collection_name: str = "nc_xml_knowledge"):
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._embedder = None
        if _HAS_CHROMA and _HAS_ST:
            self._client = chromadb.PersistentClient(
                path=settings.chroma_persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(collection_name)
            self._embedder = SentenceTransformer(settings.embedding_model)
        else:
            # Fallback: in-memory mock retrieval.
            self._embedder = _HashEmbedder()
            self._memory: list[dict] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.encode(texts)

    def build_index(self, docs: list[dict]) -> None:
        """Build/update the vector index from knowledge docs.

        docs: [{id, text, metadata}]
        """
        texts = [d["text"] for d in docs]
        ids = [d["id"] for d in docs]
        metas = [d.get("metadata", {}) for d in docs]
        embeds = self.embed(texts)

        if self._collection is not None:
            # Incremental upsert enables knowledge updates (a key reason for Chroma).
            self._collection.upsert(ids=ids, embeddings=embeds, documents=texts, metadatas=metas)
        else:
            self._memory = docs

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Similarity search, returns [{text, metadata, score}] sorted desc by score."""
        top_k = top_k or settings.rag_top_k
        qvec = self.embed([query])

        if self._collection is not None:
            res = self._collection.query(
                query_embeddings=qvec,
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            out = []
            for i in range(len(res["ids"][0])):
                dist = res["distances"][0][i]
                score = 1.0 - dist / 2.0  # distance -> similarity (cosine distance in [0,2])
                out.append(
                    {
                        "text": res["documents"][0][i],
                        "metadata": res["metadatas"][0][i],
                        "score": round(score, 4),
                    }
                )
            return [r for r in out if r["score"] >= settings.rag_score_threshold]

        # Mock fallback: brute-force cosine similarity in memory.
        # Character n-gram hash embeddings have low absolute similarity, so the
        # fallback uses a looser threshold to demonstrate "recall"; once BGE is
        # swapped in, settings.rag_score_threshold (0.35) is the sensible value.
        scored = []
        for d in self._memory:
            dv = self.embed([d["text"]])[0]
            sim = sum(a * b for a, b in zip(qvec[0], dv, strict=False))
            scored.append((sim, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"text": d["text"], "metadata": d.get("metadata", {}), "score": round(s, 4)}
            for s, d in scored[:top_k]
            if s > 0.05
        ]


# Global singleton
vector_store = VectorStore()


def load_knowledge_from_dir(directory: str) -> None:
    """Load all .json docs from a directory and build the index (called at startup)."""
    docs: list[dict] = []
    for p in Path(directory).glob("*.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        for item in data:
            docs.append(
                {
                    "id": item["id"],
                    "text": item["text"],
                    "metadata": item.get("metadata", {}),
                }
            )
    if docs:
        vector_store.build_index(docs)
