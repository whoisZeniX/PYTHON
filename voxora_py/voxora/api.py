import time

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from voxora.pipeline import run_rag
from voxora.services import Services
from voxora.stt import display_language

router = APIRouter()


def get_services(request: Request) -> Services:
    return request.app.state.services


class TextQueryRequest(BaseModel):
    query: str
    language: str = "English"


async def transcribe_upload(services: Services, audio: UploadFile) -> tuple[str, str, float]:
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="No audio was recorded. Please try again.")

    result = await run_in_threadpool(
        services.stt.transcribe, content, audio.filename or "recording.webm"
    )
    if not result.transcript:
        raise HTTPException(status_code=400, detail="No speech detected.")
    return result.transcript, display_language(result.language_code), result.latency_ms


@router.post("/api/query")
async def text_query(body: TextQueryRequest, request: Request):
    start = time.perf_counter()
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    services = get_services(request)
    return await run_in_threadpool(run_rag, services, query, body.language, 0.0, start)


@router.post("/api/voice-query")
async def voice_query(request: Request, audio: UploadFile = File(...)):
    start = time.perf_counter()
    services = get_services(request)
    transcript, language, stt_ms = await transcribe_upload(services, audio)
    return await run_in_threadpool(run_rag, services, transcript, language, stt_ms, start)


@router.get("/api/latency/summary")
async def latency_summary(request: Request, sample_size: int = 10):
    return get_services(request).telemetry.summary(sample_size)


@router.get("/healthz")
async def healthz(request: Request):
    return {"status": "ok", "documents": len(get_services(request).corpus)}
