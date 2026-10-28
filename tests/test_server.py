import json
import threading

import pytest

import llm
import server
from conftest import call
from helpers import Recorder, groq_reply, triage_json


def stored():
    return json.loads(server.DATA_PATH.read_text())


def use(monkeypatch, *responses):
    rec = Recorder(*responses)
    monkeypatch.setattr(llm.requests, "post", rec)
    return rec


# ---------------------------------------------------------------- list / get / update
def test_list_is_newest_first_and_filterable():
    everything = call(server.list_tickets)
    stamps = [t["timestamp"] for t in everything]
    assert stamps == sorted(stamps, reverse=True)
    assert all(t["status"] == "open" for t in call(server.list_tickets, status="open"))
    assert all(t["category"] == "KYC Issue" for t in call(server.list_tickets, category="kyc issue"))
    assert "error" in call(server.list_tickets, status="nonsense")[0]


def test_get_ticket():
    first = call(server.list_tickets)[0]
    assert call(server.get_ticket, first["id"])["id"] == first["id"]
    assert "error" in call(server.get_ticket, "T-0")


def test_update_status_persists_to_disk():
    tid = call(server.list_tickets)[0]["id"]
    assert call(server.update_ticket_status, tid, "closed")["status"] == "closed"
    assert next(t for t in stored() if t["id"] == tid)["status"] == "closed"
    assert "error" in call(server.update_ticket_status, tid, "paused")
    assert "error" in call(server.update_ticket_status, "T-0", "open")


# ---------------------------------------------------------------- create
def test_create_ticket_fields():
    before = len(stored())
    out = call(server.create_ticket, "  Payout short for PRT-4321 ", "Missing TDS credit", "Priya Rao", "Partner")
    t = out["ticket"]
    assert t["id"] == "T-1240" and t["source"] == "ui" and t["status"] == "open"
    assert t["category"] == "Unclassified" and t["priority"] == "unclassified"
    assert t["subject"] == "Payout short for PRT-4321" and t["sender_type"] == "partner"
    assert t["reference_id"] == "PRT-4321" and t["timestamp"].endswith("Z")
    assert len(stored()) == before + 1
    assert call(server.list_tickets)[0]["id"] == t["id"]  # newest first


@pytest.mark.parametrize("args, fragment", [
    (("", "body"), "required"), (("subject", "   "), "required"),
    (("s" * 201, "body"), "at most"), (("s", "b" * 4001), "at most"),
    (("s", "b", "x", "alien"), "sender_type"),
])
def test_create_ticket_validation(args, fragment):
    before = len(stored())
    out = call(server.create_ticket, *args)
    assert fragment in out["error"] and len(stored()) == before


def test_create_ticket_defaults_blank_sender():
    assert call(server.create_ticket, "s", "b", "  ")["ticket"]["sender"] == "Anonymous"


def test_queue_cap(monkeypatch):
    monkeypatch.setattr(server, "MAX_TICKETS", len(stored()))
    assert "queue is full" in call(server.create_ticket, "s", "b")["error"]


def test_exact_copy_of_existing_ticket_is_flagged_as_duplicate():
    original = next(t for t in stored() if t.get("reference_id") and t["source"] == "seed")
    out = call(server.create_ticket, original["subject"], original["body"], original["sender"], original["sender_type"])
    assert original["id"] in {d["id"] for d in out["possible_duplicates"]}
    assert out["ticket"]["id"] not in {d["id"] for d in out["possible_duplicates"]}


def test_unrelated_new_ticket_is_not_flagged():
    out = call(server.create_ticket, "Zebra migration", "Quarterly wildlife census of the northern plains", "Zed")
    assert out["possible_duplicates"] == []


def test_concurrent_creates_get_unique_ids():
    before = len(stored())
    ids, errors = [], []

    def worker(i):
        try:
            ids.append(call(server.create_ticket, f"s{i}", f"body number {i}", f"user{i}")["ticket"]["id"])
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(set(ids)) == 12 and len(stored()) == before + 12


def test_find_duplicates_tool():
    dup = next(t for t in stored() if t.get("duplicate_of") and t.get("reference_id"))
    out = call(server.find_duplicates, dup["id"])
    assert dup["duplicate_of"] in {d["id"] for d in out["possible_duplicates"]}
    assert "error" in call(server.find_duplicates, "T-0")


# ---------------------------------------------------------------- classify
def test_classify_seeded_ticket_never_overwrites_labels(monkeypatch):
    rec = use(monkeypatch, groq_reply(triage_json(category="KYC Issue", priority="low")))
    t = stored()[5]
    out = call(server.classify_ticket, t["id"], True)  # apply requested, but the ticket is seeded
    assert out["applied"] is False and out["category"] == "KYC Issue"
    again = next(x for x in stored() if x["id"] == t["id"])
    assert again["category"] == t["category"] and again["priority"] == t["priority"] and "triage" not in again
    assert "REFERENCE TICKETS" in rec.user_prompt  # retrieved examples were grounded into the prompt


def test_classify_examples_are_resolved_and_labeled(monkeypatch):
    rec = use(monkeypatch, groq_reply(triage_json()))
    call(server.classify_ticket, stored()[0]["id"])
    ids = call(server.classify_ticket, stored()[0]["id"])["similar_ticket_ids"]
    by_id = {t["id"]: t for t in stored()}
    assert 1 <= len(ids) <= 3 and all(by_id[i]["status"] != "open" and by_id[i]["category"] in server.CATEGORIES for i in ids)
    assert rec.payload["response_format"] == {"type": "json_object"}


def test_apply_saves_prediction_on_ui_ticket(monkeypatch):
    use(monkeypatch, groq_reply(triage_json(category="Disbursement Delay", priority="high", sentiment="angry", reply="On it.")))
    tid = call(server.create_ticket, "Money stuck", "Disbursement pending for BHL-99999", "Rohit")["ticket"]["id"]
    out = call(server.classify_ticket, tid, True)
    assert out["applied"] is True and out["ticket"]["category"] == "Disbursement Delay"
    saved = next(t for t in stored() if t["id"] == tid)
    assert saved["priority"] == "high" and saved["triage"]["suggested_response"] == "On it."
    assert saved["triage"]["sentiment"] == "angry" and saved["triage"]["model"] == "openai/gpt-oss-20b"


def test_apply_skipped_when_prediction_is_unusable(monkeypatch):
    use(monkeypatch, groq_reply(triage_json(category="Refunds")))
    tid = call(server.create_ticket, "s", "b")["ticket"]["id"]
    out = call(server.classify_ticket, tid, True)
    assert out["applied"] is False and out["category_valid"] is False
    assert next(t for t in stored() if t["id"] == tid)["category"] == "Unclassified"


def test_classify_without_apply_leaves_ui_ticket_alone(monkeypatch):
    use(monkeypatch, groq_reply(triage_json()))
    tid = call(server.create_ticket, "s", "b")["ticket"]["id"]
    assert call(server.classify_ticket, tid)["applied"] is False
    assert next(t for t in stored() if t["id"] == tid)["category"] == "Unclassified"


def test_classify_error_paths(monkeypatch):
    assert "not found" in call(server.classify_ticket, "T-0")["error"]
    use(monkeypatch, groq_reply("not json at all"))
    out = call(server.classify_ticket, stored()[0]["id"])
    assert "parse" in out["error"] and out["raw_output"] == "not json at all"
    monkeypatch.delenv("GROQ_API_KEY")
    assert "GROQ_API_KEY" in call(server.classify_ticket, stored()[0]["id"])["error"]


# ---------------------------------------------------------------- ask
def test_ask_tickets(monkeypatch):
    rec = use(monkeypatch, groq_reply("Several cases are high priority."))
    out = call(server.ask_tickets, "Which cases are about KYC mismatch?")
    assert out["answer"] == "Several cases are high priority." and out["referenced_ticket_ids"]
    assert "response_format" not in rec.payload  # free text, not JSON mode
    assert "error" in call(server.ask_tickets, "   ")
