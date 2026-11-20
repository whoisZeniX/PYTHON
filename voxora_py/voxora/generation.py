import json
import time
from typing import Any

import groq
from fastapi import HTTPException

SYSTEM_PROMPT = (
    "You are Voxora, a helpful assistant. Answer the user's question clearly in a few sentences. "
    "If the supplied context contains the answer, answer from it without inventing facts, "
    'set source to "documents" and cite the document IDs you used. '
    "If the context does not contain the answer, answer from your own general knowledge, "
    'set source to "general_knowledge" and return an empty citations list. '
    "Answer in the same language as the question. "
    "Return only JSON."
)

ANSWER_SCHEMA = {
    "name": "grounded_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "string"}},
            "grounded": {"type": "boolean"},
            "confidence": {"type": "number"},
            "source": {"type": "string", "enum": ["documents", "general_knowledge"]},
        },
        "required": ["answer", "citations", "grounded", "confidence", "source"],
        "additionalProperties": False,
    },
}


def build_context(documents: list[dict[str, Any]]) -> str:
    blocks = [
        f"[DOCUMENT {n}]\ndocument_id: {doc.get('document_id', '')}\ncontent: {doc.get('english_text') or doc.get('text', '')}"
        for n, doc in enumerate(documents, start=1)
    ]
    return "\n\n---\n\n".join(blocks) if blocks else "NO CONTEXT AVAILABLE"


def build_user_message(query: str, context: str) -> str:
    return (
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        "Return JSON with this exact structure:\n"
        '{"answer":"...","citations":["document_id"],"grounded":true,"confidence":0.0,"source":"documents"}'
    )


class GroqGenerator:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_completion_tokens: int = 512,
        timeout_s: float = 60.0,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.client = (
            groq.Groq(api_key=api_key, timeout=timeout_s, max_retries=max_retries) if api_key else None
        )

    def generate(self, query: str, documents: list[dict[str, Any]]) -> tuple[dict[str, Any], float]:
        if self.client is None:
            raise HTTPException(status_code=502, detail="Groq API key is not configured")

        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(query, build_context(documents))},
                ],
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
                response_format={"type": "json_schema", "json_schema": ANSWER_SCHEMA},
            )
        except (groq.AuthenticationError, groq.PermissionDeniedError):
            raise HTTPException(status_code=502, detail="Groq API key is invalid")
        except groq.APITimeoutError:
            raise HTTPException(status_code=504, detail="Groq request timed out")
        except groq.RateLimitError:
            raise HTTPException(status_code=502, detail="Groq rate limit reached")
        except groq.APIConnectionError:
            raise HTTPException(status_code=502, detail="Could not reach Groq")
        except groq.BadRequestError as err:
            if "json_validate_failed" in str(err.body):
                raise HTTPException(status_code=502, detail="Groq could not produce a valid answer, please try again")
            raise HTTPException(status_code=502, detail="Groq request failed (400)")
        except groq.APIStatusError as err:
            raise HTTPException(status_code=502, detail=f"Groq request failed ({err.status_code})")
        generation_ms = round((time.perf_counter() - start) * 1000.0, 2)

        content = response.choices[0].message.content or ""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = None
        if not isinstance(data, dict):
            raise HTTPException(status_code=502, detail="Groq returned an invalid response")
        return data, generation_ms
