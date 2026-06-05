import os
import shutil
import time

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import UPLOAD_DIR
from pipe_line import ingest_document, chat
from models import ChatRequest, ChatResponse
from logger import get_logger

log = get_logger("app")
app = FastAPI(title="RAG FastAPI Project", version="1.0.0")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    log.info("REQUEST  %s %s", request.method, request.url.path)
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    log.info("RESPONSE %s %s -> %d (%.1f ms)",
             request.method, request.url.path, response.status_code, elapsed)
    return response


## allow all the coro orgins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    log.debug("Health check called")
    return {"status": "ok", "message": "RAG API is running"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    log.info("Upload started: filename=%s", file.filename)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    log.info("File saved to disk: %s", file_path)

    try:
        num_chunks = ingest_document(file_path)
    except ValueError as e:
        log.warning("Unsupported file type: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Ingestion failed for %s: %s", file.filename, e)
        raise HTTPException(status_code=500, detail="Ingestion failed")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            log.info("Temp file removed: %s", file_path)

    log.info("Upload complete: filename=%s chunks=%d", file.filename, num_chunks)
    return {"filename": file.filename, "chunks_indexed": num_chunks}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    log.info("Chat request: question=%r k=%d", req.question, req.k)
    try:
        answer, sources = chat(req.question, req.k)
    except Exception as e:
        log.exception("Chat failed: %s", e)
        raise HTTPException(status_code=500, detail="Chat failed")
    log.info("Chat response: sources=%s", sources)
    return ChatResponse(answer=answer, sources=sources)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000,reload=True)