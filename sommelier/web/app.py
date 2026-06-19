"""FastAPI application startup."""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from sommelier.web.api import router

WEB_DIR = Path(__file__).parent


def create_app() -> FastAPI:
    """Create and configure the web application."""

    app = FastAPI(title="AI Sommelier Assistant")
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        """Serve the chat page."""

        return (WEB_DIR / "templates" / "index.html").read_text(encoding="utf-8")

    return app


app = create_app()
