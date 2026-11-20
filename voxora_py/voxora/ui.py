import time
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from voxora.api import get_services, transcribe_upload
from voxora.config import ROOT
from voxora.pipeline import run_rag
from voxora.sessions import THEME_COOKIE, is_dark

router = APIRouter()
templates = Jinja2Templates(directory=ROOT / "templates")


SUGGESTED_QUESTIONS = [
    "What is Hack Club?",
    "What is GitHub?",
    "What is Docker?",
    "What is FastAPI?",
    "What is Python?",
]

FEATURES = [
    {
        "title": "Retrieval-Augmented Generation",
        "desc": "Automatically extracts dense embeddings from the knowledge base to ensure the AI's answer is deeply grounded in factual evidence, effectively reducing hallucinations.",
        "icon": "M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10",
        "color": "bg-yellow-300",
    },
    {
        "title": "Live Audio Visualizer",
        "desc": "Intercepts the browser's Web Audio API stream and renders the exact frequencies of your voice directly onto a dynamic HTML5 canvas.",
        "icon": "M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z",
        "color": "bg-blue-300",
    },
    {
        "title": "Real-time Telemetry",
        "desc": "Collects precise execution latency (P50, P70, P100 metrics) for Speech-to-Text, Embedding generation, and LLM inference across your sessions.",
        "icon": "M13 10V3L4 14h7v7l9-11h-7z",
        "color": "bg-red-300",
    },
    {
        "title": "Session History Drawer",
        "desc": "Maintains a full log of your current workspace session. Instantly retrieve past voice queries, exact transcripts, and their corresponding AI answers.",
        "icon": "M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4",
        "color": "bg-green-300",
    },
]


def css_version() -> int:
    path = ROOT / "static" / "app.css"
    return int(path.stat().st_mtime) if path.exists() else 0


def render(request: Request, name: str, page: str = "app", status_code: int = 200, **context: Any) -> HTMLResponse:
    services = get_services(request)
    session = services.sessions.get(request)
    base = {
        "page": page,
        "dark": is_dark(request),
        "history": list(session.history),
        "summary": services.telemetry.summary(services.settings.latency_sample_size),
        "suggestions": SUGGESTED_QUESTIONS,
        "css_version": css_version(),
        "result": None,
        "error": None,
    }
    base.update(context)
    return templates.TemplateResponse(request, name, base, status_code=status_code)


@router.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return render(request, "pages/landing.html", page="landing")


@router.get("/features", response_class=HTMLResponse)
async def features(request: Request):
    return render(request, "pages/features.html", page="features", features=FEATURES)


@router.get("/app", response_class=HTMLResponse)
async def workspace(request: Request):
    current = get_services(request).sessions.get(request).current
    return render(request, "pages/workspace.html", page="app", result=current)


def show_result(request: Request, result: dict, query_value: str | None = None) -> HTMLResponse:
    get_services(request).sessions.add_result(request, result)
    return render(request, "partials/swap.html", result=result, query_value=query_value)


def show_error(request: Request, err: HTTPException, query_value: str | None = None) -> HTMLResponse:
    get_services(request).sessions.clear_current(request)
    return render(
        request, "partials/swap.html", error=str(err.detail), query_value=query_value, status_code=err.status_code
    )


@router.post("/ui/query", response_class=HTMLResponse)
async def ui_query(request: Request, query: str = Form(""), source: str = Form("form")):
    start = time.perf_counter()
    query = query.strip()
    if not query:
        return Response(status_code=204)

    chip_text = query if source == "chip" else None
    try:
        result = await run_in_threadpool(run_rag, get_services(request), query, "English", 0.0, start)
    except HTTPException as err:
        return show_error(request, err, chip_text)
    return show_result(request, result, chip_text)


@router.post("/ui/voice", response_class=HTMLResponse)
async def ui_voice(request: Request, audio: UploadFile = File(...)):
    start = time.perf_counter()
    services = get_services(request)
    try:
        transcript, language, stt_ms = await transcribe_upload(services, audio)
        result = await run_in_threadpool(run_rag, services, transcript, language, stt_ms, start)
    except HTTPException as err:
        return show_error(request, err)
    return show_result(request, result)


@router.get("/ui/history/{number}", response_class=HTMLResponse)
async def ui_history(request: Request, number: int):
    result = get_services(request).sessions.select(request, number)
    if result is None:
        return render(request, "partials/result_area.html", error="History item not found.", status_code=404)
    return render(request, "partials/result_area.html", result=result)


@router.get("/ui/theme/toggle")
async def toggle_theme(request: Request, next: str = "/"):
    target = next if next.startswith("/") and not next.startswith("//") else "/"
    response = RedirectResponse(target, status_code=303)
    response.set_cookie(
        THEME_COOKIE,
        "light" if is_dark(request) else "dark",
        max_age=60 * 60 * 24 * 365,
        samesite="lax",
    )
    return response
