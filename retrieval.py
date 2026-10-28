"""
Ticket retrieval and duplicate detection on top of embeddings + a vector store.

Used for three things:
  * similar()    few-shot context for classification (resolved, labeled tickets only)
  * relevant()   grounding for free-text questions (any ticket)
  * duplicates() flag a new ticket that repeats an existing one

The index is self-healing: every public method first embeds any ticket it hasn't seen
(`sync`), so tickets created at runtime, or edited on disk, are picked up automatically.
"""

import os
import re
import threading
from typing import Callable, Dict, List, Optional, Set, Tuple

from embeddings import get_embedder
from vectorstore import VectorStore

ID_RE = re.compile(r"\b(?:BHL|PRT)-\d{4,6}\b", re.IGNORECASE)

# Similarity thresholds depend on the embedder (a hashing vector and a neural model score
# the same pair very differently). These are starting points: run
# `python evaluate.py duplicates` with your real embedder and set DUPLICATE_THRESHOLD.
DEFAULT_THRESHOLDS = {"hashing": 0.25, "fastembed": 0.88}  # hashing: measured; fastembed: an uncalibrated guess
ID_MATCH_OFFSET = 0.20      # same loan/partner ID => accept a lower similarity
SAME_SENDER_OFFSET = 0.25   # same sender (no ID conflict) => accept a lower similarity
NEAR_COPY_FLOOR = 0.60      # anonymous senders and no shared ID: only a (near) copy counts, whatever the threshold
GENERIC_SENDERS = {"", "anonymous", "customer", "partner", "unknown"}

Ticket = Dict[str, object]


def ticket_text(t: Ticket) -> str:
    return ". ".join(p for p in (str(t.get("subject", "")).strip(), str(t.get("body", "")).strip()) if p)


def extract_ids(text: str) -> Set[str]:
    return {m.upper() for m in ID_RE.findall(text)}


def default_threshold(embedder) -> float:
    if os.environ.get("DUPLICATE_THRESHOLD"):
        return float(os.environ["DUPLICATE_THRESHOLD"])
    return DEFAULT_THRESHOLDS.get(getattr(embedder, "kind", ""), 0.80)


def duplicate_verdict(sim: float, ids_a: Set[str], ids_b: Set[str], sender_a: str, sender_b: str,
                      threshold: float) -> Optional[str]:
    """Why two tickets look like duplicates, or None. Shared by the app and the evaluation."""
    if ids_a and ids_b:
        if not (ids_a & ids_b):
            return None  # different loan / partner => a different case, however similar the wording
        return "same loan/partner ID with related wording" if sim >= threshold - ID_MATCH_OFFSET else None
    a, b = sender_a.strip().lower(), sender_b.strip().lower()
    named_a, named_b = a not in GENERIC_SENDERS, b not in GENERIC_SENDERS
    if named_a and named_b:
        if a != b:
            return None  # two different named people with no shared ID are two different cases
        return "same sender with related wording" if sim >= threshold - SAME_SENDER_OFFSET else None
    return "near-identical wording" if sim >= max(threshold, NEAR_COPY_FLOOR) else None


class TicketIndex:
    def __init__(self, embedder=None):
        self.embedder = embedder or get_embedder()
        self.store = VectorStore(self.embedder.dim)
        self._embed_lock = threading.Lock()  # embedding models are not guaranteed re-entrant

    def _embed(self, texts: List[str]):
        with self._embed_lock:
            return self.embedder.embed(texts)

    def sync(self, tickets: List[Ticket]) -> None:
        missing = [t for t in tickets if t["id"] not in self.store]
        if missing:
            self.store.add([str(t["id"]) for t in missing], self._embed([ticket_text(t) for t in missing]))

    def _rank(self, text: str, tickets: List[Ticket], k: int,
              accept: Optional[Callable[[Ticket], bool]] = None) -> List[Tuple[Ticket, float]]:
        self.sync(tickets)
        by_id = {str(t["id"]): t for t in tickets}
        hits = self.store.search(self._embed([text])[0], k,
                                 predicate=lambda i: i in by_id and (accept is None or accept(by_id[i])))
        return [(by_id[i], s) for i, s in hits]

    def similar(self, ticket: Ticket, tickets: List[Ticket], k: int = 3,
                accept: Optional[Callable[[Ticket], bool]] = None,
                exclude_ids: Set[str] = frozenset()) -> List[Tuple[Ticket, float]]:
        """Nearest other tickets to `ticket`, optionally restricted by `accept`."""
        return self._rank(
            ticket_text(ticket), tickets, k,
            lambda t: t["id"] != ticket.get("id") and t["id"] not in exclude_ids and (accept is None or accept(t)))

    def relevant(self, query: str, tickets: List[Ticket], k: int = 8) -> List[Tuple[Ticket, float]]:
        return [(t, s) for t, s in self._rank(query, tickets, k) if s > 0]

    def duplicates(self, ticket: Ticket, tickets: List[Ticket], threshold: Optional[float] = None,
                   k: int = 3) -> List[dict]:
        threshold = default_threshold(self.embedder) if threshold is None else threshold
        text, ids = ticket_text(ticket), extract_ids(ticket_text(ticket))
        out = []
        for other, sim in self._rank(text, tickets, len(tickets), lambda t: t["id"] != ticket.get("id")):
            reason = duplicate_verdict(sim, ids, extract_ids(ticket_text(other)),
                                       str(ticket.get("sender", "")), str(other.get("sender", "")), threshold)
            if reason:
                out.append({"id": other["id"], "subject": other.get("subject", ""), "status": other.get("status"),
                            "similarity": round(sim, 3), "reason": reason})
        return out[:k]


_index: Optional[TicketIndex] = None


def get_index() -> TicketIndex:
    global _index
    if _index is None:
        _index = TicketIndex()
    return _index


def reset_index() -> None:
    global _index
    _index = None
