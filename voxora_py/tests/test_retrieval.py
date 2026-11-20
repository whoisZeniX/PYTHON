import faiss
import numpy as np

from voxora.retrieval import BM25Index, Retriever, tokenize

CORPUS = [
    {"document_id": f"d{i}", "text": text, "source": "test", "language": "eng_Latn"}
    for i, text in enumerate(
        [
            "docker containers ship software",
            "python is a programming language",
            "the manhattan project built the bomb",
            "github hosts git repositories",
            "fastapi is a python web framework",
            "hack club is a community of teen hackers",
        ]
    )
]


class StubEmbedder:

    def __init__(self, vector):
        self.vector = np.asarray([vector], dtype=np.float32)

    def embed_query(self, text):
        return self.vector


VECTORS = np.random.default_rng(0).normal(size=(len(CORPUS), 4)).astype(np.float32)
VECTORS /= np.linalg.norm(VECTORS, axis=1, keepdims=True)


def make_retriever(query_vector, min_dense_score=0.30):
    index = faiss.IndexFlatIP(4)
    index.add(VECTORS)
    bm25 = BM25Index.build([d["text"] for d in CORPUS])
    return Retriever(StubEmbedder(query_vector), index, index, bm25, CORPUS, min_dense_score=min_dense_score)


def test_tokenize_strips_ascii_and_indic_punctuation():
    assert tokenize("Hello, World! यह है। ठीक॥") == ["hello", "world", "यह", "है", "ठीक"]
    assert tokenize("") == []


def test_dense_results_when_score_is_high():
    retriever = make_retriever(VECTORS[2])
    result = retriever.retrieve("anything")
    assert not result.fallback_used
    assert len(result.documents) == 5
    assert result.documents[0]["document_id"] == "d2"
    assert [d["rank"] for d in result.documents] == [1, 2, 3, 4, 5]
    assert {d["retrieval_type"] for d in result.documents} == {"dense"}


def test_low_dense_score_falls_back_to_bm25():
    retriever = make_retriever([0, 0, 0, 0])
    result = retriever.retrieve("python web framework")
    assert result.fallback_used
    assert result.documents[0]["document_id"] == "d4"
    assert {d["retrieval_type"] for d in result.documents} == {"bm25"}
    assert [d["rank"] for d in result.documents] == list(range(1, len(result.documents) + 1))
    assert len(result.documents) <= 5


def test_fallback_with_no_bm25_tokens_returns_nothing():
    retriever = make_retriever([0, 0, 0, 0])
    result = retriever.retrieve("!!! ???")
    assert result.fallback_used
    assert result.documents == []


def test_real_index_fallback_with_raised_threshold(real_services):
    retriever = real_services.retriever
    normal = retriever.retrieve("What is Docker?")
    assert not normal.fallback_used and len(normal.documents) == 5

    original = retriever.min_dense_score
    retriever.min_dense_score = 0.99
    try:
        forced = retriever.retrieve("मैनहट्टन परियोजना")
    finally:
        retriever.min_dense_score = original
    assert forced.fallback_used
    assert {d["retrieval_type"] for d in forced.documents} == {"bm25"}
