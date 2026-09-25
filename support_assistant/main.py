"""
FastAPI wrapper around the LangGraph assistant.

    uvicorn main:app --host 0.0.0.0 --port 7860
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from ingest import ensure_index
from supportassistant import AskRequest, AskResponse, ask, mock_llm_enabled


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_index()   # build the ChromaDB index on first start if it isn't there yet
    yield


app = FastAPI(title="Zepto Support Assistant", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "mock_llm": mock_llm_enabled()}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    return ask(request.query)
