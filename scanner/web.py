"""Web app: upload or paste anything, get a report. Run with `python -m scanner serve`."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from . import config, scan_bytes, scan_text
from .core import REGISTRY

app = FastAPI(title="AI Scanner")
STATIC = Path(__file__).parent / "static"


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/analyzers")
def list_analyzers():
    return [{"name": s.name, "kinds": s.kinds, "description": s.description} for s in REGISTRY]


@app.post("/api/scan")
async def scan(file: UploadFile | None = File(None), text: str | None = Form(None)):
    if file is not None and file.filename:
        data = await file.read()
        if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"File is larger than {config.MAX_UPLOAD_MB} MB")
        return await run_in_threadpool(scan_bytes, data, file.filename)
    if text and text.strip():
        return await run_in_threadpool(scan_text, text)
    raise HTTPException(400, "Send a file or some text")
