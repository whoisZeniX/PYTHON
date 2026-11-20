import time
from typing import Any

from voxora.guardrails import UNSAFE_ANSWER, is_unsafe, validate_grounding


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 2)


def _response(
    query: str,
    language: str,
    answer: str,
    status: str,
    timings: dict[str, float],
    documents: list[dict[str, Any]] | None = None,
    citations: list[str] | None = None,
    grounded: bool = False,
    fallback_used: bool = False,
    grounding_confidence: float = 0.0,
) -> dict[str, Any]:
    return {
        "transcript": query,
        "language": language,
        "answer": answer,
        "grounded": grounded,
        "citations": citations or [],
        "status": status,
        "timings": timings,
        "retrieved_documents": [
            {key: value for key, value in doc.items() if key != "english_text"} for doc in documents or []
        ],
        "fallback_used": fallback_used,
        "grounding_confidence": grounding_confidence,
    }


def run_rag(services, query: str, language: str, stt_ms: float, start: float) -> dict[str, Any]:
    if is_unsafe(query):
        timings = {
            "stt_ms": stt_ms,
            "retrieval_ms": 0.0,
            "generation_ms": 0.0,
            "grounding_ms": 0.0,
            "total_rag_ms": 0.0,
            "total_e2e_ms": _elapsed_ms(start),
        }
        return _response(query, language, UNSAFE_ANSWER, "refused_unsafe", timings)

    retrieval = services.retriever.retrieve(query)
    documents = retrieval.documents

    generated, generation_ms = services.generator.generate(query, documents)

    grounding_start = time.perf_counter()
    answer = str(generated.get("answer") or "")
    citations = [str(c) for c in generated.get("citations") or []]
    general = generated.get("source") == "general_knowledge" or not citations or not documents
    if general:
        citations, grounded, confidence = [], False, 0.0
        status = "general_knowledge"
    else:
        grounded, confidence = validate_grounding(
            answer, citations, documents, services.settings.min_grounding_overlap
        )
        status = "success" if grounded else "ungrounded"
    grounding_ms = _elapsed_ms(grounding_start)

    timings = {
        "stt_ms": stt_ms,
        "retrieval_ms": retrieval.retrieval_ms,
        "generation_ms": generation_ms,
        "grounding_ms": grounding_ms,
        "total_rag_ms": round(retrieval.retrieval_ms + generation_ms + grounding_ms, 2),
        "total_e2e_ms": _elapsed_ms(start),
    }
    services.telemetry.add(timings)

    return _response(
        query,
        language,
        answer,
        status,
        timings,
        documents=documents,
        citations=citations,
        grounded=grounded,
        fallback_used=retrieval.fallback_used,
        grounding_confidence=confidence,
    )
