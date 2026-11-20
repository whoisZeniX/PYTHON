import time
from dataclasses import dataclass

import requests
from fastapi import HTTPException

CONTENT_TYPES = {
    "wav": "audio/wav",
    "webm": "audio/webm",
    "ogg": "audio/ogg",
    "mp3": "audio/mpeg",
    "m4a": "audio/mp4",
    "aac": "audio/aac",
    "flac": "audio/flac",
}


@dataclass
class Transcription:
    transcript: str
    language_code: str
    latency_ms: float


def display_language(language_code: str) -> str:
    return "Hindi" if "hi" in (language_code or "").lower() else "English"


class SarvamSTT:
    def __init__(self, api_key: str, url: str, model: str = "saaras:v4", timeout_s: float = 60.0):
        self.api_key = api_key
        self.url = url
        self.model = model
        self.timeout_s = timeout_s

    def transcribe(self, audio: bytes, filename: str = "recording.webm") -> Transcription:
        if not self.api_key:
            raise HTTPException(status_code=502, detail="Sarvam API key is not configured")

        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else "webm"
        content_type = CONTENT_TYPES.get(extension, "application/octet-stream")

        start = time.perf_counter()
        try:
            response = requests.post(
                self.url,
                headers={"api-subscription-key": self.api_key},
                data={"model": self.model, "language_code": "unknown", "mode": "transcribe"},
                files={"file": (filename, audio, content_type)},
                timeout=self.timeout_s,
            )
        except requests.Timeout:
            raise HTTPException(status_code=504, detail="Sarvam request timed out")
        except requests.RequestException:
            raise HTTPException(status_code=502, detail="Could not reach Sarvam")
        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)

        if response.status_code in (401, 403):
            raise HTTPException(status_code=502, detail="Sarvam API key is invalid")
        if not response.ok:
            raise HTTPException(
                status_code=502,
                detail=f"Sarvam STT request failed ({response.status_code}): {_error_message(response)}",
            )

        result = response.json()
        return Transcription(
            transcript=(result.get("transcript") or "").strip(),
            language_code=result.get("language_code") or "unknown",
            latency_ms=latency_ms,
        )


def _error_message(response: requests.Response) -> str:
    try:
        error = response.json().get("error")
    except ValueError:
        error = None
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    return response.text[:200]
