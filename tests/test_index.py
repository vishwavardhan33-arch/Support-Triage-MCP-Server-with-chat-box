import json

import retrieval
import server
from retrieval import TicketIndex, duplicate_verdict, extract_ids, get_index, ticket_text


def tickets():
    return json.loads(server.DATA_PATH.read_text())


def test_extract_ids_finds_loan_and_partner_ids():
    assert extract_ids("see bhl-12345 and PRT-9876, not ABC-1") == {"BHL-12345", "PRT-9876"}


def test_ticket_text_handles_missing_subject():
    assert ticket_text({"subject": "", "body": "just a body"}) == "just a body"


def test_sync_is_incremental():
    idx, data = TicketIndex(), tickets()
    idx.sync(data)
    assert len(idx.store) == len(data)
    idx.sync(data + [{"id": "T-new", "subject": "x", "body": "new ticket", "status": "open"}])
    assert len(idx.store) == len(data) + 1


def test_similar_excludes_self_and_respects_filters():
    idx, data = TicketIndex(), tickets()
    target = data[0]
    hits = idx.similar(target, data, k=5, accept=lambda t: t["status"] != "open")
    assert hits and all(t["id"] != target["id"] and t["status"] != "open" for t, _ in hits)
    banned = {hits[0][0]["id"]}
    again = idx.similar(target, data, k=5, accept=lambda t: t["status"] != "open", exclude_ids=banned)
    assert banned.isdisjoint(t["id"] for t, _ in again)


def test_relevant_finds_matching_ticket_first():
    idx, data = TicketIndex(), tickets()
    probe = next(t for t in data if t["category"] == "Interest Rate Query")
    hits = idx.relevant(probe["body"], data, k=3)
    assert hits[0][0]["id"] == probe["id"] and all(s > 0 for _, s in hits)


def test_labeled_id_duplicates_are_flagged():
    idx, data = TicketIndex(), tickets()
    by_id = {t["id"]: t for t in data}
    pairs = [(t, by_id[t["duplicate_of"]]) for t in data if t.get("duplicate_of") and t.get("reference_id")]
    assert len(pairs) == 16
    found = sum(1 for dup, orig in pairs if orig["id"] in {d["id"] for d in idx.duplicates(dup, data, k=10)})
    assert found == len(pairs)


def test_followup_without_id_is_linked_by_sender():
    idx, data = TicketIndex(), tickets()
    by_id = {t["id"]: t for t in data}
    followups = [t for t in data if t.get("duplicate_of") and not t.get("reference_id")]
    found = sum(1 for dup in followups
                if by_id[dup["duplicate_of"]]["id"] in {d["id"] for d in idx.duplicates(dup, data, k=10)})
    assert found >= len(followups) - 2  # bag-of-words can miss a heavily reworded follow-up


def test_duplicate_result_shape_and_ordering():
    idx, data = TicketIndex(), tickets()
    dup = next(t for t in data if t.get("duplicate_of") and t.get("reference_id"))
    out = idx.duplicates(dup, data, k=5)
    assert {"id", "subject", "status", "similarity", "reason"} <= set(out[0])
    sims = [d["similarity"] for d in out]
    assert sims == sorted(sims, reverse=True) and all(d["id"] != dup["id"] for d in out)


def test_verdict_rules():
    thr = 0.5
    a, b = {"BHL-1"}, {"BHL-2"}
    assert duplicate_verdict(0.99, a, b, "x", "x", thr) is None                      # different cases
    assert "same loan" in duplicate_verdict(0.31, a, a, "x", "y", thr)               # shared ID, lower bar
    assert duplicate_verdict(0.29, a, a, "x", "y", thr) is None
    assert duplicate_verdict(0.99, set(), set(), "Priya Rao", "Amit Joshi", thr) is None   # two named people
    assert "same sender" in duplicate_verdict(0.26, set(), set(), "Priya Rao", "priya rao", thr)
    assert "near-identical" in duplicate_verdict(0.95, set(), set(), "Anonymous", "Amit Joshi", thr)
    assert duplicate_verdict(0.55, set(), set(), "Anonymous", "Amit Joshi", thr) is None
    assert duplicate_verdict(0.55, set(), set(), "Anonymous", "Amit Joshi", 0.3) is None   # floor of 0.60 applies


def test_threshold_override_from_env(monkeypatch):
    monkeypatch.setenv("DUPLICATE_THRESHOLD", "0.42")
    assert retrieval.default_threshold(get_index().embedder) == 0.42


def test_index_survives_concurrent_adds_and_searches(monkeypatch):
    """Regression: an unlocked FAISS index segfaulted when one thread added while others searched."""
    import threading
    data = tickets()[:40]
    idx = TicketIndex()
    idx.sync(data)
    errors, stop = [], threading.Event()

    def adder(k):
        try:
            for i in range(30):
                extra = {"id": f"X{k}-{i}", "subject": "new", "body": f"extra ticket {k} {i} about funds", "status": "open"}
                idx.sync(data + [extra])
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def searcher():
        try:
            while not stop.is_set():
                idx.relevant("funds not received", data, k=5)
                idx.similar(data[0], data, k=3)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    adders = [threading.Thread(target=adder, args=(k,)) for k in range(4)]
    searchers = [threading.Thread(target=searcher) for _ in range(4)]
    [t.start() for t in adders + searchers]
    [t.join() for t in adders]
    stop.set()
    [t.join() for t in searchers]
    assert not errors
    assert len(idx.store) == len(set(idx.store.ids)) == 40 + 4 * 30
