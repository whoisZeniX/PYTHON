import re
from typing import Any

UNSAFE_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all instructions",
    r"system prompt",
    r"reveal your prompt",
    r"jailbreak",
    r"how to build a bomb",
    r"how to make explosives",
    r"how to hack into",
]

UNSAFE_ANSWER = "I can't help with that request."

WORD_PATTERN = re.compile(r"[a-zA-Z]{4,}")


def is_unsafe(query: str) -> bool:
    text = query.lower()
    return any(re.search(pattern, text) for pattern in UNSAFE_PATTERNS)


def validate_grounding(
    answer: str,
    citations: list[str],
    documents: list[dict[str, Any]],
    min_overlap: float = 0.10,
) -> tuple[bool, float]:
    valid_ids = {str(doc["document_id"]) for doc in documents}

    if not citations or not all(str(c) in valid_ids for c in citations):
        return False, 0.0

    answer_words = set(WORD_PATTERN.findall(answer.lower()))
    if not answer_words:
        return True, 1.0

    context_words: set[str] = set()
    for doc in documents:
        context_words.update(WORD_PATTERN.findall((doc.get("text") or "").lower()))
        context_words.update(WORD_PATTERN.findall((doc.get("english_text") or "").lower()))

    overlap = len(answer_words & context_words) / len(answer_words)
    return overlap >= min_overlap, round(overlap, 4)
