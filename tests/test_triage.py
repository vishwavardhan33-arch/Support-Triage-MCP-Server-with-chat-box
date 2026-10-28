import pytest

import llm
import triage
from helpers import Recorder, groq_reply, triage_json

TICKET = {"id": "T-1", "subject": "Funds missing", "body": "Nothing credited for BHL-11111", "sender_type": "partner"}


def use(monkeypatch, *responses):
    rec = Recorder(*responses)
    monkeypatch.setattr(llm.requests, "post", rec)
    return rec


def test_system_prompt_lists_categories_and_guards_against_injection():
    p = triage.build_system_prompt(["A", "B"], "a test desk")
    assert "exactly one of: A, B" in p and "a test desk" in p and "untrusted" in p


def test_user_prompt_includes_labeled_examples_and_clips_long_text():
    ex = [{"id": "T-2", "subject": "s", "body": "x" * 500, "category": "KYC Issue", "priority": "medium"},
          {"id": "T-3", "subject": "", "body": "short", "category": "Transfers"}]
    prompt = triage.build_user_prompt(TICKET, ex)
    assert "Subject: Funds missing" in prompt and "REFERENCE TICKETS" in prompt
    assert "category: KYC Issue, priority: medium" in prompt and "category: Transfers" in prompt
    assert "x" * 300 not in prompt
    assert "REFERENCE" not in triage.build_user_prompt(TICKET, [])


def test_classify_happy_path_normalises_labels(monkeypatch):
    use(monkeypatch, groq_reply(triage_json(category=" disbursement  delay ", priority="HIGH", sentiment="Angry")))
    out = triage.classify(TICKET, [{"id": "T-9", "subject": "", "body": "b", "category": "KYC Issue"}])
    assert out["category"] == "Disbursement Delay" and out["category_valid"] is True
    assert out["priority"] == "high" and out["sentiment"] == "angry"
    assert out["similar_ticket_ids"] == ["T-9"]


def test_invalid_labels_are_flagged_not_hidden(monkeypatch):
    use(monkeypatch, groq_reply(triage_json(category="Refund Problems", priority="urgent", sentiment="meh")))
    out = triage.classify(TICKET)
    assert out["category"] == "Refund Problems" and out["category_valid"] is False
    assert out["priority"] is None and out["sentiment"] is None


def test_custom_category_list(monkeypatch):
    rec = use(monkeypatch, groq_reply(triage_json(category="Transfers")))
    out = triage.classify(TICKET, categories=["Transfers", "Top-ups"], domain="a bank")
    assert out["category_valid"] is True
    assert "Transfers, Top-ups" in rec.payload["messages"][0]["content"]


def test_fenced_json_is_accepted(monkeypatch):
    use(monkeypatch, groq_reply("```json\n" + triage_json() + "\n```"))
    assert triage.classify(TICKET)["category"] == "Disbursement Delay"


def test_unparseable_output_keeps_raw_text(monkeypatch):
    use(monkeypatch, groq_reply("I cannot do that"))
    with pytest.raises(triage.TriageParseError) as exc:
        triage.classify(TICKET)
    assert exc.value.raw == "I cannot do that"


def test_llm_errors_propagate(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(llm.LLMError):
        triage.classify(TICKET)


def test_answer_question_summarises_queue(monkeypatch):
    rec = use(monkeypatch, groq_reply("  Two cases are open.  "))
    tickets = [{"id": "T-1", "status": "open", "priority": "high", "subject": "a", "body": "b"},
               {"id": "T-2", "status": "closed", "priority": "low", "subject": "c", "body": "d"}]
    assert triage.answer_question("How many open?", tickets, tickets[:1]) == "Two cases are open."
    assert "2 cases" in rec.user_prompt and "'open': 1" in rec.user_prompt and "T-1" in rec.user_prompt
    assert "response_format" not in rec.payload
