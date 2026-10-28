import json

import pytest
from fastapi.testclient import TestClient

import app as app_module
import llm
import server
from helpers import Recorder, groq_reply, triage_json

client = TestClient(app_module.app)


def use(monkeypatch, *responses):
    rec = Recorder(*responses)
    monkeypatch.setattr(llm.requests, "post", rec)
    return rec


def first_id():
    return client.get("/api/tickets").json()[0]["id"]


def test_health_reports_llm_and_search_stack():
    body = client.get("/api/health").json()
    assert body["llm"] == {"provider": "groq", "model": "openai/gpt-oss-20b", "configured": True}
    assert body["embeddings"].startswith("hashing") and body["vector_store"] in {"faiss", "numpy"}


def test_lifespan_warms_the_index():
    with TestClient(app_module.app) as warm:
        assert warm.get("/api/health").json()["indexed_tickets"] == len(json.loads(server.DATA_PATH.read_text()))


def test_list_filter_and_get():
    assert len(client.get("/api/tickets").json()) >= 200
    assert all(t["status"] == "open" for t in client.get("/api/tickets?status=open").json())
    assert client.get("/api/tickets?status=bogus").status_code == 400
    assert client.get(f"/api/tickets/{first_id()}").status_code == 200
    assert client.get("/api/tickets/NOPE").status_code == 404


def test_update_status():
    tid = first_id()
    assert client.patch(f"/api/tickets/{tid}/status", json={"status": "closed"}).json()["status"] == "closed"
    assert client.patch(f"/api/tickets/{tid}/status", json={"status": "x"}).status_code == 400
    assert client.patch("/api/tickets/NOPE/status", json={"status": "open"}).status_code == 404


def test_create_ticket_returns_201_with_duplicates_field():
    r = client.post("/api/tickets", json={"subject": "Salary slip upload fails", "body": "Error on upload for BHL-70001",
                                          "sender": "Anita", "sender_type": "customer"})
    assert r.status_code == 201
    body = r.json()
    assert body["ticket"]["source"] == "ui" and body["ticket"]["id"].startswith("T-") and "possible_duplicates" in body
    assert client.get(f"/api/tickets/{body['ticket']['id']}").status_code == 200


@pytest.mark.parametrize("payload", [
    {"subject": "", "body": "b"}, {"subject": "   ", "body": "b"}, {"subject": "s", "body": ""},
    {"subject": "s" * 201, "body": "b"}, {"subject": "s", "body": "b" * 4001},
    {"subject": "s", "body": "b", "sender_type": "robot"}, {"subject": "s", "body": "b", "sender": "n" * 81}, {},
])
def test_create_ticket_rejects_bad_input(payload):
    assert client.post("/api/tickets", json=payload).status_code == 422


def test_queue_full_is_409(monkeypatch):
    monkeypatch.setattr(server, "MAX_TICKETS", 1)
    assert client.post("/api/tickets", json={"subject": "s", "body": "b"}).status_code == 409


def test_new_ticket_text_is_stored_verbatim_not_interpreted():
    nasty = "<script>alert(1)</script> {{7*7}} ' OR 1=1 --"
    r = client.post("/api/tickets", json={"subject": nasty, "body": nasty, "sender": nasty[:80]})
    assert r.status_code == 201 and r.json()["ticket"]["subject"] == nasty


def test_duplicate_of_existing_ticket_is_reported_on_create():
    original = next(t for t in client.get("/api/tickets").json() if t.get("reference_id") and t["source"] == "seed")
    r = client.post("/api/tickets", json={"subject": original["subject"], "body": original["body"],
                                          "sender": original["sender"], "sender_type": original["sender_type"]})
    assert original["id"] in {d["id"] for d in r.json()["possible_duplicates"]}
    assert client.get(f"/api/tickets/{r.json()['ticket']['id']}/duplicates").json()["possible_duplicates"]
    assert client.get("/api/tickets/NOPE/duplicates").status_code == 404


def test_classify_with_and_without_apply(monkeypatch):
    use(monkeypatch, groq_reply(triage_json(category="Account Access", priority="medium")))
    tid = client.post("/api/tickets", json={"subject": "Login", "body": "OTP never arrives"}).json()["ticket"]["id"]
    plain = client.post(f"/api/tickets/{tid}/classify").json()
    assert plain["category"] == "Account Access" and plain["applied"] is False
    assert client.get(f"/api/tickets/{tid}").json()["category"] == "Unclassified"
    applied = client.post(f"/api/tickets/{tid}/classify?apply=true").json()
    assert applied["applied"] is True
    stored = client.get(f"/api/tickets/{tid}").json()
    assert stored["category"] == "Account Access" and stored["triage"]["model"] == "openai/gpt-oss-20b"


def test_classify_errors(monkeypatch):
    assert client.post("/api/tickets/NOPE/classify").status_code == 404
    monkeypatch.delenv("GROQ_API_KEY")
    assert client.post(f"/api/tickets/{first_id()}/classify").status_code == 502


def test_ask(monkeypatch):
    use(monkeypatch, groq_reply("Five cases are open."))
    r = client.post("/api/ask", json={"question": "How many are open?"})
    assert r.status_code == 200 and r.json()["answer"] == "Five cases are open."
    assert client.post("/api/ask", json={"question": "x" * 501}).status_code == 422
    assert client.post("/api/ask", json={"question": "  "}).status_code == 502


def test_rate_limit_per_client(monkeypatch):
    monkeypatch.setattr(app_module.per_ip, "limit", 2)
    body = {"subject": "s", "body": "b"}
    a, b = {"x-forwarded-for": "9.9.9.9, 1.1.1.1"}, {"x-forwarded-for": "9.9.9.9, 2.2.2.2"}
    assert client.post("/api/tickets", json=body, headers=a).status_code == 201
    assert client.post("/api/tickets", json=body, headers=a).status_code == 201
    assert client.post("/api/tickets", json=body, headers=a).status_code == 429
    assert client.post("/api/tickets", json=body, headers=b).status_code == 201  # last hop identifies the client


def test_global_rate_limit_is_a_backstop(monkeypatch):
    monkeypatch.setattr(app_module.global_limit, "limit", 2)
    body = {"subject": "s", "body": "b"}
    assert client.post("/api/tickets", json=body, headers={"x-forwarded-for": "1.1.1.1"}).status_code == 201
    assert client.post("/api/tickets", json=body, headers={"x-forwarded-for": "2.2.2.2"}).status_code == 201
    assert client.post("/api/tickets", json=body, headers={"x-forwarded-for": "3.3.3.3"}).status_code == 429


def test_reads_are_not_rate_limited(monkeypatch):
    monkeypatch.setattr(app_module.per_ip, "limit", 1)
    assert all(client.get("/api/tickets").status_code == 200 for _ in range(5))


def test_frontend_is_served_with_new_case_flow():
    html = client.get("/").text
    assert "New case" in html and "Ask about your cases" in html
    assert "newCaseForm" in client.get("/app.js").text
