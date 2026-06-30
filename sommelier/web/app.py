"""FastAPI application startup."""

import asyncio
import os
from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from sommelier.storage.session_repository import SessionRepository
from sommelier.web.api import router
from sommelier.web.llm_keepalive import keepalive_loop, warmup_llm_safely

WEB_DIR = Path(__file__).parent


def _keepalive_seconds(configured: float | None) -> float:
    if configured is not None:
        return max(0.0, configured)
    raw = os.environ.get("SOMMELIER_LLM_KEEPALIVE_SECONDS", "0")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def create_app(
    repository: SessionRepository | None = None,
    *,
    keepalive_seconds: float | None = None,
    keepalive_model_factory: Callable[[], Any] | None = None,
) -> FastAPI:
    """Create and configure the web application."""

    interval = _keepalive_seconds(keepalive_seconds)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task: asyncio.Task | None = None
        if interval > 0:
            app.state.llm_warmup_succeeded = await warmup_llm_safely(
                keepalive_model_factory
            )
            task = asyncio.create_task(
                keepalive_loop(interval, keepalive_model_factory)
            )
        else:
            app.state.llm_warmup_succeeded = None
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(title="AI Sommelier Assistant", lifespan=lifespan)
    app.state.repository = repository
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        """Serve the chat page."""

        return (WEB_DIR / "templates" / "index.html").read_text(encoding="utf-8")

    return app


app = create_app()
