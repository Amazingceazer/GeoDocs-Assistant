import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="GeoDocs Assistant", version="0.1.0")
app.include_router(router)

# Mounted last so API routes take precedence; serves the demo UI at /
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")
