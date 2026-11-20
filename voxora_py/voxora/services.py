import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import faiss
import pyarrow.parquet as pq
from fastapi import FastAPI

from voxora.config import Settings, get_settings
from voxora.embedder import E5Embedder
from voxora.generation import GroqGenerator
from voxora.retrieval import BM25Index, Retriever
from voxora.sessions import SessionStore
from voxora.stt import SarvamSTT
from voxora.telemetry import LatencyStore

logger = logging.getLogger("voxora")

CORPUS_COLUMNS = ["document_id", "text", "language", "source", "english_text"]


@dataclass
class Services:
    settings: Settings
    corpus: list[dict[str, Any]]
    retriever: Retriever
    generator: GroqGenerator
    stt: SarvamSTT
    telemetry: LatencyStore
    sessions: SessionStore


def load_corpus(settings: Settings) -> list[dict[str, Any]]:
    return pq.read_table(settings.corpus_path, columns=CORPUS_COLUMNS).to_pylist()


def load_services(settings: Settings | None = None) -> Services:
    settings = settings or get_settings()

    corpus = load_corpus(settings)
    logger.info("Corpus loaded: %s documents", f"{len(corpus):,}")

    faiss_index = faiss.read_index(str(settings.faiss_path))
    faiss_en_index = faiss.read_index(str(settings.faiss_en_path))
    bm25 = BM25Index.load(settings.bm25_path)
    logger.info(
        "Indexes loaded: FAISS hi %s x %s, FAISS en %s x %s, BM25 %s documents",
        faiss_index.ntotal, faiss_index.d, faiss_en_index.ntotal, faiss_en_index.d, bm25.corpus_size,
    )

    for name, size in (("FAISS hi", faiss_index.ntotal), ("FAISS en", faiss_en_index.ntotal), ("BM25", bm25.corpus_size)):
        if size != len(corpus):
            raise RuntimeError(f"{name} has {size} rows but the corpus has {len(corpus)}. Rebuild it with scripts/.")

    embedder = E5Embedder(settings.onnx_model_path, settings.tokenizer_path, settings.embed_max_tokens)
    logger.info("Embedder loaded: multilingual-e5-small (ONNX int8, CPU)")

    retriever = Retriever(
        embedder,
        faiss_index,
        faiss_en_index,
        bm25,
        corpus,
        search_top_k=settings.search_top_k,
        final_top_k=settings.final_top_k,
        min_dense_score=settings.min_dense_score,
    )
    generator = GroqGenerator(
        settings.groq_api_key,
        settings.groq_model,
        temperature=settings.temperature,
        max_completion_tokens=settings.max_completion_tokens,
        timeout_s=settings.groq_timeout_s,
        max_retries=settings.groq_max_retries,
    )
    stt = SarvamSTT(
        settings.sarvam_api_key, settings.sarvam_url, settings.sarvam_model, settings.sarvam_timeout_s
    )
    logger.info(
        "Groq model %s (key %s); Sarvam %s (key %s)",
        settings.groq_model,
        "set" if settings.groq_api_key else "MISSING",
        settings.sarvam_model,
        "set" if settings.sarvam_api_key else "MISSING",
    )

    return Services(
        settings=settings,
        corpus=corpus,
        retriever=retriever,
        generator=generator,
        stt=stt,
        telemetry=LatencyStore(),
        sessions=SessionStore(settings.history_limit),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "services", None) is None:
        app.state.services = load_services()
    logger.info("Voxora ready")
    yield
