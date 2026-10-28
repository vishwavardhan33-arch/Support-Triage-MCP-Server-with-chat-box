import pytest
import requests

import llm
from helpers import Recorder, Resp, groq_reply


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)


def test_groq_request_shape(monkeypatch):
    rec = Recorder(groq_reply('{"a": 1}'))
    monkeypatch.setattr(llm.requests, "post", rec)
    out = llm.call_llm("sys", "user", json_mode=True)
    assert out == '{"a": 1}'
    call = rec.calls[0]
    assert call["url"] == llm.GROQ_URL
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert call["json"]["model"] == "openai/gpt-oss-20b"
    assert call["json"]["temperature"] == 0
    assert call["json"]["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in call["json"]["messages"]] == ["system", "user"]


def test_free_text_mode_has_no_response_format(monkeypatch):
    rec = Recorder(groq_reply("plain"))
    monkeypatch.setattr(llm.requests, "post", rec)
    llm.call_llm("s", "u", json_mode=False)
    assert "response_format" not in rec.payload


def test_model_override_and_reasoning_effort(monkeypatch):
    monkeypatch.setenv("GROQ_REASONING_EFFORT", "low")
    rec = Recorder(groq_reply("x"))
    monkeypatch.setattr(llm.requests, "post", rec)
    llm.call_llm("s", "u", model="openai/gpt-oss-120b")
    assert rec.payload["model"] == "openai/gpt-oss-120b"
    assert rec.payload["reasoning_effort"] == "low"


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(llm.LLMError, match="GROQ_API_KEY"):
        llm.call_llm("s", "u")
    assert llm.llm_info()["configured"] is False


def test_rate_limit_is_retried(monkeypatch):
    rec = Recorder(Resp(429, {}, headers={"retry-after": "0"}), Resp(429, {}), groq_reply("ok"))
    monkeypatch.setattr(llm.requests, "post", rec)
    assert llm.call_llm("s", "u") == "ok"
    assert len(rec.calls) == 3


def test_rate_limit_gives_up(monkeypatch):
    rec = Recorder(Resp(429, {}))
    monkeypatch.setattr(llm.requests, "post", rec)
    with pytest.raises(llm.LLMError, match="429"):
        llm.call_llm("s", "u")
    assert len(rec.calls) == llm.MAX_RETRIES + 1


def test_json_mode_rejected_falls_back_to_prompt_only(monkeypatch):
    rec = Recorder(Resp(400, {}, text='{"error": "response_format is not supported"}'), groq_reply("ok"))
    monkeypatch.setattr(llm.requests, "post", rec)
    assert llm.call_llm("s", "u") == "ok"
    assert "response_format" not in rec.payload


def test_http_error_and_network_error(monkeypatch):
    monkeypatch.setattr(llm.requests, "post", Recorder(Resp(401, {}, text="bad key")))
    with pytest.raises(llm.LLMError, match="401"):
        llm.call_llm("s", "u")
    monkeypatch.setattr(llm.requests, "post", Recorder(requests.exceptions.ConnectionError("down")))
    with pytest.raises(llm.LLMError, match="Could not reach Groq"):
        llm.call_llm("s", "u")


def test_unexpected_response_shape(monkeypatch):
    monkeypatch.setattr(llm.requests, "post", Recorder(Resp(200, {"nope": []})))
    with pytest.raises(llm.LLMError, match="shape"):
        llm.call_llm("s", "u")


def test_ollama_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    rec = Recorder(Resp(200, {"response": "hello"}))
    monkeypatch.setattr(llm.requests, "post", rec)
    assert llm.call_llm("s", "u") == "hello"
    assert rec.payload["format"] == "json" and rec.payload["model"] == "llama3.2"
    assert llm.llm_info() == {"provider": "ollama", "model": "llama3.2", "configured": True}
    monkeypatch.setattr(llm.requests, "post", Recorder(requests.exceptions.ConnectionError()))
    with pytest.raises(llm.LLMError, match="Ollama"):
        llm.call_llm("s", "u")


def test_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "carrier-pigeon")
    with pytest.raises(llm.LLMError, match="Unknown"):
        llm.call_llm("s", "u")


@pytest.mark.parametrize("text", ['{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! Here you go: {"a": 1} Hope that helps.'])
def test_parse_json_object_tolerates_wrapping(text):
    assert llm.parse_json_object(text) == {"a": 1}


@pytest.mark.parametrize("text", ["no json here", "[1, 2, 3]", ""])
def test_parse_json_object_rejects_garbage(text):
    with pytest.raises(ValueError):
        llm.parse_json_object(text)
