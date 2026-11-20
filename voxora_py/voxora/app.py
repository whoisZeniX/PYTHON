import logging
import secrets

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from voxora import api, ui
from voxora.config import ROOT, get_settings
from voxora.services import lifespan

logger = logging.getLogger("voxora")


def create_app(services=None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Voxora", lifespan=lifespan)
    app.state.services = services

    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret or secrets.token_urlsafe(32),
        session_cookie="voxora_session",
        same_site="lax",
    )
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    app.include_router(api.router)
    app.include_router(ui.router)

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s", request.url.path)
        detail = f"Internal error: {type(exc).__name__}"
        if request.url.path.startswith("/ui/"):
            return ui.render(request, "partials/result_area.html", error=detail, status_code=500)
        return JSONResponse({"detail": detail}, status_code=500)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        if request.url.path.startswith("/ui/"):
            return ui.render(request, "partials/result_area.html", error=str(exc.detail), status_code=exc.status_code)
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)

    return app
