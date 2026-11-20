import pickle
import re
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

INDIC_PUNCTUATION = string.punctuation + "।॥''\"“”‘’"
PUNCT_PATTERN = re.compile("[" + re.escape(INDIC_PUNCTUATION) + "]")


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return PUNCT_PATTERN.sub(" ", text.lower().strip()).split()


class BM25Index:
    def __init__(self, bm25: BM25Okapi, corpus_size: int, k1: float = 1.5, b: float = 0.75):
        self.bm25 = bm25
        self.corpus_size = corpus_size
        self.k1 = k1
        self.b = b

    @classmethod
    def build(cls, texts: list[str], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        return cls(BM25Okapi([tokenize(t) for t in texts], k1=k1, b=b), len(texts), k1, b)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        return cls(data["bm25"], data["corpus_size"], data["k1"], data["b"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"bm25": self.bm25, "k1": self.k1, "b": self.b, "corpus_size": self.corpus_size}
        with open(path, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def search(self, query: str, top_k: int) -> list[tuple[float, int]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens).astype(np.float32)
        indices = np.argsort(scores)[::-1][: min(top_k, self.corpus_size)]
        return [(float(scores[i]), int(i)) for i in indices]


def dense_search(index: faiss.Index, vector: np.ndarray, top_k: int) -> list[tuple[float, int]]:
    vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    scores, indices = index.search(vector, min(top_k, index.ntotal))
    return [(float(s), int(i)) for s, i in zip(scores[0], indices[0]) if i >= 0]


@dataclass
class RetrievalResult:
    documents: list[dict[str, Any]]
    fallback_used: bool
    retrieval_ms: float


class Retriever:
    def __init__(
        self,
        embedder,
        faiss_index: faiss.Index,
        faiss_en_index: faiss.Index,
        bm25: BM25Index,
        corpus: list[dict[str, Any]],
        search_top_k: int = 10,
        final_top_k: int = 5,
        min_dense_score: float = 0.30,
    ):
        self.embedder = embedder
        self.faiss_index = faiss_index
        self.faiss_en_index = faiss_en_index
        self.bm25 = bm25
        self.corpus = corpus
        self.search_top_k = search_top_k
        self.final_top_k = final_top_k
        self.min_dense_score = min_dense_score

    def _document(self, index: int, score: float, retrieval_type: str) -> dict[str, Any]:
        row = self.corpus[index]
        return {
            "document_id": str(row.get("document_id", "")),
            "text": row.get("text") or "",
            "score": score,
            "rank": 0,
            "source": row.get("source") or "",
            "language": row.get("language") or "",
            "retrieval_type": retrieval_type,
            "english_text": row.get("english_text") or "",
        }

    def _dense_hits(self, query: str) -> list[tuple[float, int]]:
        vector = self.embedder.embed_query(query)
        hits = dense_search(self.faiss_index, vector, self.search_top_k)
        hits += dense_search(self.faiss_en_index, vector, self.search_top_k)
        best: dict[int, float] = {}
        for score, index in hits:
            best[index] = max(score, best.get(index, score))
        ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
        return [(score, index) for index, score in ranked[: self.search_top_k]]

    def retrieve(self, query: str) -> RetrievalResult:
        start = time.perf_counter()

        hits = self._dense_hits(query)
        retrieval_type = "dense"
        fallback_used = False

        top_score = hits[0][0] if hits else 0.0
        if top_score < self.min_dense_score:
            hits = self.bm25.search(query, self.search_top_k)
            retrieval_type = "bm25"
            fallback_used = True

        documents = [self._document(i, s, retrieval_type) for s, i in hits[: self.final_top_k]]
        for rank, doc in enumerate(documents, start=1):
            doc["rank"] = rank

        retrieval_ms = round((time.perf_counter() - start) * 1000.0, 2)
        return RetrievalResult(documents, fallback_used, retrieval_ms)
