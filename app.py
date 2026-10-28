"""
Web UI backend for the support-triage project.

Wraps the same tool functions the MCP server exposes (server.py) as a REST API and serves
the static frontend. Run locally: python app.py  ->  http://localhost:8000
On Render: uvicorn app:app --host 0.0.0.0 --port $PORT
"""

import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

import llm
import server
from embeddings import get_embedder
from retrieval import get_index


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the embedding model and embed the queue once at startup, not on the first request.
    get_index().sync(server._load_tickets())
    yield


app = FastAPI(title="Support Triage", lifespan=lifespan)


def _fn(tool):
    """Tools decorated with @mcp.tool() may be wrapped; .fn is the plain function."""
    return getattr(tool, "fn", tool)


# ---------------------------------------------------------------- abuse protection
class RateLimiter:
    """Sliding-window counter. In-memory, so per process: fine for a single-instance demo."""

    def __init__(self, limit: int, window: float = 60.0):
        self.limit, self.window = limit, window
        self.hits = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            q = self.hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                raise HTTPException(429, "Too many requests right now. Please wait a minute and try again.")
            q.append(now)


per_ip = RateLimiter(int(os.environ.get("RATE_LIMIT_PER_MIN", "10")))
global_limit = RateLimiter(int(os.environ.get("RATE_LIMIT_GLOBAL_PER_MIN", "60")))


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:  # the last hop is the one appended by the hosting proxy; earlier ones can be spoofed
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def throttle(request: Request) -> None:
    """Applied to endpoints that write data or spend LLM quota."""
    global_limit.check("global")
    per_ip.check(client_ip(request))


# ---------------------------------------------------------------- models
class StatusUpdate(BaseModel):
    status: str


class AskRequest(BaseModel):
    question: str = Field(max_length=500)


class NewTicket(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    subject: str = Field(min_length=1, max_length=server.MAX_SUBJECT)
    body: str = Field(min_length=1, max_length=server.MAX_BODY)
    sender: str = Field(default="Anonymous", max_length=80)
    sender_type: Literal["customer", "partner"] = "customer"


def _raise_for(result: dict, default_status: int) -> None:
    if "error" in result:
        message = result["error"]
        code = 404 if "not found" in message else 409 if "queue is full" in message else default_status
        raise HTTPException(code, message)


# ---------------------------------------------------------------- routes
@app.get("/api/health")
def health():
    index = get_index()
    return {"llm": llm.llm_info(), "embeddings": get_embedder().name,
            "vector_store": index.store.backend, "indexed_tickets": len(index.store)}


@app.get("/api/tickets")
def api_list_tickets(status: Optional[str] = None, category: Optional[str] = None):
    result = _fn(server.list_tickets)(status=status, category=category)
    if result and isinstance(result[0], dict) and "error" in result[0]:
        raise HTTPException(400, result[0]["error"])
    return result


@app.post("/api/tickets", status_code=201)
async def api_create_ticket(body: NewTicket, request: Request):
    throttle(request)
    result = await run_in_threadpool(_fn(server.create_ticket), body.subject, body.body, body.sender, body.sender_type)
    _raise_for(result, 400)
    return result


@app.get("/api/tickets/{ticket_id}")
def api_get_ticket(ticket_id: str):
    result = _fn(server.get_ticket)(ticket_id)
    _raise_for(result, 404)
    return result


@app.patch("/api/tickets/{ticket_id}/status")
def api_update_status(ticket_id: str, body: StatusUpdate):
    result = _fn(server.update_ticket_status)(ticket_id, body.status)
    _raise_for(result, 400)
    return result


@app.get("/api/tickets/{ticket_id}/duplicates")
def api_duplicates(ticket_id: str):
    result = _fn(server.find_duplicates)(ticket_id)
    _raise_for(result, 404)
    return result


@app.post("/api/tickets/{ticket_id}/classify")
async def api_classify(ticket_id: str, request: Request, apply: bool = False):
    throttle(request)
    result = await run_in_threadpool(_fn(server.classify_ticket), ticket_id, apply)
    _raise_for(result, 502)
    return result


@app.post("/api/ask")
async def api_ask(body: AskRequest, request: Request):
    throttle(request)
    result = await run_in_threadpool(_fn(server.ask_tickets), body.question)
    _raise_for(result, 502)
    return result


# Mounted last so the /api/* routes above always win over static files.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
