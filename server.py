"""
Support Triage MCP Server
--------------------------
Exposes a support-ticket queue as MCP tools so any MCP client can work with it.

Tools:
    list_tickets(status?, category?)       newest first
    get_ticket(ticket_id)
    update_ticket_status(ticket_id, status)
    create_ticket(subject, body, sender?, sender_type?)   also returns possible duplicates
    find_duplicates(ticket_id)
    classify_ticket(ticket_id, apply?)     LLM triage grounded in similar resolved tickets
    ask_tickets(question)                  free-text Q&A over the queue

Storage is a JSON file (data/tickets.json) guarded by a lock with atomic writes. On hosts
with an ephemeral disk (e.g. Render's free tier) changes are lost on restart/redeploy.
"""

import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mcp.server.mcpserver import MCPServer

import llm
import triage
from retrieval import extract_ids, get_index

DATA_PATH = Path(os.environ.get("TICKETS_PATH", Path(__file__).parent / "data" / "tickets.json"))
VALID_STATUSES = {"open", "in_progress", "closed"}
VALID_SENDER_TYPES = {"customer", "partner"}
CATEGORIES = triage.HOME_LOAN_CATEGORIES
UNCLASSIFIED = "Unclassified"
MAX_TICKETS = int(os.environ.get("MAX_TICKETS", "600"))
MAX_SUBJECT, MAX_BODY = 200, 4000

mcp = MCPServer("Support Triage")
_lock = threading.RLock()


# ---------------------------------------------------------------- storage
REPLACE_RETRIES = 8


def _load_tickets() -> list[dict]:
    with _lock:  # never read while another thread is swapping the file in (Windows refuses that)
        with DATA_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)


def _save_tickets(tickets: list[dict]) -> None:
    tmp = DATA_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(tickets, indent=2), encoding="utf-8")
    for attempt in range(REPLACE_RETRIES):
        try:
            os.replace(tmp, DATA_PATH)  # atomic: readers never see a half-written file
            return
        except PermissionError:
            # Windows only: antivirus, the search indexer or a backup tool can hold the file open for a
            # few milliseconds, and replacing an open file raises "Access is denied". Retry briefly.
            if attempt == REPLACE_RETRIES - 1:
                raise
            time.sleep(0.02 * (attempt + 1))


def _find(tickets: list[dict], ticket_id: str) -> Optional[dict]:
    return next((t for t in tickets if t["id"] == ticket_id), None)


# ---------------------------------------------------------------- tools
@mcp.tool()
def list_tickets(status: Optional[str] = None, category: Optional[str] = None) -> list[dict]:
    """List support tickets (newest first), optionally filtered by status and/or category."""
    tickets = _load_tickets()
    if status:
        if status not in VALID_STATUSES:
            return [{"error": f"Invalid status '{status}'. Must be one of {sorted(VALID_STATUSES)}"}]
        tickets = [t for t in tickets if t["status"] == status]
    if category:
        tickets = [t for t in tickets if str(t.get("category", "")).lower() == category.lower()]
    return sorted(tickets, key=lambda t: t.get("timestamp", ""), reverse=True)


@mcp.tool()
def get_ticket(ticket_id: str) -> dict:
    """Get full details of a single ticket by its ID (e.g. "T-1021")."""
    return _find(_load_tickets(), ticket_id) or {"error": f"Ticket '{ticket_id}' not found"}


@mcp.tool()
def update_ticket_status(ticket_id: str, status: str) -> dict:
    """Update a ticket's status (open, in_progress or closed)."""
    if status not in VALID_STATUSES:
        return {"error": f"Invalid status '{status}'. Must be one of {sorted(VALID_STATUSES)}"}
    with _lock:
        tickets = _load_tickets()
        ticket = _find(tickets, ticket_id)
        if ticket is None:
            return {"error": f"Ticket '{ticket_id}' not found"}
        ticket["status"] = status
        _save_tickets(tickets)
        return ticket


@mcp.tool()
def create_ticket(subject: str, body: str, sender: str = "Anonymous", sender_type: str = "customer") -> dict:
    """
    File a new ticket. It starts open and Unclassified; run classify_ticket(apply=True) to triage it.
    Returns the ticket plus any possible duplicates among existing tickets.
    """
    subject, body = (subject or "").strip(), (body or "").strip()
    sender = (sender or "").strip()[:80] or "Anonymous"
    sender_type = (sender_type or "").strip().lower()
    if not subject or not body:
        return {"error": "Both a subject and a message are required"}
    if len(subject) > MAX_SUBJECT or len(body) > MAX_BODY:
        return {"error": f"Subject must be at most {MAX_SUBJECT} characters and the message at most {MAX_BODY}"}
    if sender_type not in VALID_SENDER_TYPES:
        return {"error": f"sender_type must be one of {sorted(VALID_SENDER_TYPES)}"}

    with _lock:
        tickets = _load_tickets()
        if len(tickets) >= MAX_TICKETS:
            return {"error": f"This demo queue is full ({MAX_TICKETS} tickets)"}
        numbers = [int(t["id"][2:]) for t in tickets if re.fullmatch(r"T-\d+", t["id"])]
        ids = sorted(extract_ids(f"{subject} {body}"))
        ticket = {
            "id": f"T-{max(numbers, default=999) + 1}", "subject": subject, "body": body, "sender": sender,
            "sender_type": sender_type, "reference_id": ids[0] if ids else None,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "category": UNCLASSIFIED, "priority": "unclassified", "status": "open", "source": "ui",
        }
        tickets.append(ticket)
        _save_tickets(tickets)
    return {"ticket": ticket, "possible_duplicates": get_index().duplicates(ticket, tickets)}


@mcp.tool()
def find_duplicates(ticket_id: str) -> dict:
    """List existing tickets that look like duplicates of this one, with a similarity score and reason."""
    tickets = _load_tickets()
    ticket = _find(tickets, ticket_id)
    if ticket is None:
        return {"error": f"Ticket '{ticket_id}' not found"}
    return {"ticket_id": ticket_id, "possible_duplicates": get_index().duplicates(ticket, tickets)}


@mcp.tool()
def classify_ticket(ticket_id: str, apply: bool = False) -> dict:
    """
    Triage a ticket with the LLM: category, priority, sentiment and a drafted reply, grounded in
    the most similar resolved tickets (embedding retrieval). With apply=True the result is saved
    onto tickets created via create_ticket; seeded tickets keep their ground-truth labels.
    """
    tickets = _load_tickets()
    ticket = _find(tickets, ticket_id)
    if ticket is None:
        return {"error": f"Ticket '{ticket_id}' not found"}

    examples = [t for t, _ in get_index().similar(
        ticket, tickets, k=3, accept=lambda t: t["status"] != "open" and t.get("category") in CATEGORIES)]
    try:
        result = triage.classify(ticket, examples, CATEGORIES)
    except llm.LLMError as e:
        return {"error": str(e)}
    except triage.TriageParseError as e:
        return {"error": "Could not parse model output as JSON", "raw_output": e.raw}

    result.update(ticket_id=ticket_id, model=llm.model_name(), applied=False)
    if apply and ticket.get("source") == "ui" and result["category_valid"] and result["priority"]:
        with _lock:
            tickets = _load_tickets()
            stored = _find(tickets, ticket_id)
            if stored is not None:
                stored.update(
                    category=result["category"], priority=result["priority"],
                    triage={"sentiment": result["sentiment"], "suggested_response": result["suggested_response"],
                            "model": result["model"], "similar_ticket_ids": result["similar_ticket_ids"],
                            "classified_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
                _save_tickets(tickets)
                result.update(applied=True, ticket=stored)
    return result


@mcp.tool()
def ask_tickets(question: str) -> dict:
    """Answer a free-form question about the queue, grounded in queue statistics and the most relevant tickets."""
    if not question or not question.strip():
        return {"error": "Question cannot be empty"}
    tickets = _load_tickets()
    relevant = [t for t, _ in get_index().relevant(question.strip(), tickets, k=8)]
    try:
        answer = triage.answer_question(question.strip(), tickets, relevant)
    except llm.LLMError as e:
        return {"error": str(e)}
    return {"answer": answer, "referenced_ticket_ids": [t["id"] for t in relevant]}


if __name__ == "__main__":
    mcp.run()
