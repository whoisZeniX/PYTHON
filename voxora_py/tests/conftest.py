import dataclasses
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voxora.app import create_app
from voxora.services import load_services
from voxora.sessions import SessionStore
from voxora.stt import Transcription
from voxora.telemetry import LatencyStore


class FakeGenerator:

    def __init__(self, answer: str | None = None, cite_invalid: bool = False, general: bool = False):
        self.calls = 0
        self.answer = answer
        self.cite_invalid = cite_invalid
        self.general = general

    def generate(self, query, documents):
        self.calls += 1
        answer = self.answer if self.answer is not None else "Docker packages applications into containers."
        if self.general or not documents:
            return {"answer": answer, "citations": [], "grounded": False, "confidence": 0.0, "source": "general_knowledge"}, 12.5
        citation = "not-a-real-id" if self.cite_invalid else documents[0]["document_id"]
        return {"answer": answer, "citations": [citation], "grounded": True, "confidence": 0.9, "source": "documents"}, 12.5


class FakeSTT:
    def __init__(self, transcript="What is Docker?", language_code="en-IN"):
        self.transcript = transcript
        self.language_code = language_code
        self.calls = 0

    def transcribe(self, audio: bytes, filename: str = "recording.webm"):
        self.calls += 1
        return Transcription(self.transcript, self.language_code, 321.0)


@pytest.fixture(scope="session")
def real_services():
    return load_services()


@pytest.fixture
def services(real_services):
    return dataclasses.replace(
        real_services,
        generator=FakeGenerator(),
        stt=FakeSTT(),
        telemetry=LatencyStore(),
        sessions=SessionStore(real_services.settings.history_limit),
    )


@pytest.fixture
def client(services):
    with TestClient(create_app(services)) as test_client:
        yield test_client
