"""
RAG Agent: Documents -> Embedding -> Vector store -> Retriever -> LLM -> Answer.

Uses a tiny in-memory bag-of-words "embedding" and cosine similarity so it
runs with zero dependencies. Swap SimpleEmbedder / InMemoryVectorStore for
FAISS, Chroma, Pinecone, etc. in production.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..llm import LLMBackend


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class SimpleEmbedder:
    """Bag-of-words vector as {token: count}. Not a real embedding model."""

    def embed(self, text: str) -> Dict[str, int]:
        return dict(Counter(_tokenize(text)))


def _cosine(a: Dict[str, int], b: Dict[str, int]) -> float:
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values())) or 1e-9
    nb = math.sqrt(sum(v * v for v in b.values())) or 1e-9
    return dot / (na * nb)


@dataclass
class Document:
    doc_id: str
    text: str


class InMemoryVectorStore:
    def __init__(self, embedder: SimpleEmbedder):
        self._embedder = embedder
        self._docs: List[Document] = []
        self._vectors: List[Dict[str, int]] = []

    def add(self, doc_id: str, text: str) -> None:
        self._docs.append(Document(doc_id, text))
        self._vectors.append(self._embedder.embed(text))

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        qvec = self._embedder.embed(query)
        scored = [
            (doc, _cosine(qvec, vec))
            for doc, vec in zip(self._docs, self._vectors)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [pair for pair in scored[:top_k] if pair[1] > 0]


class RAGAgent:
    def __init__(self, llm: LLMBackend, store: InMemoryVectorStore = None):
        self._llm = llm
        self._embedder = SimpleEmbedder()
        self._store = store or InMemoryVectorStore(self._embedder)

    def add_document(self, doc_id: str, text: str) -> None:
        self._store.add(doc_id, text)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        results = self._store.search(query, top_k=top_k)
        return [doc.text for doc, _score in results]

    def answer(self, query: str) -> str:
        chunks = self.retrieve(query)
        context = "\n---\n".join(chunks) if chunks else "(no relevant documents found)"
        prompt = (
            "Synthesize a final answer for the user using everything gathered so far.\n"
            f"Original request: {query}\n"
            f"Working context:\n{context}"
        )
        return self._llm.complete(prompt)
