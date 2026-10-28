import json

import requests


class Resp:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status=200, body=None, headers=None, text=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = text if text is not None else json.dumps(body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def groq_reply(content: str) -> Resp:
    return Resp(200, {"choices": [{"message": {"content": content}}]})


def triage_json(category="Disbursement Delay", priority="high", sentiment="frustrated", reply="We are escalating this now."):
    return json.dumps({"category": category, "priority": priority, "sentiment": sentiment, "suggested_response": reply})


class Recorder:
    """Callable replacement for requests.post that records every call and returns queued responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        resp = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(resp, Exception):
            raise resp
        return resp

    @property
    def payload(self):
        return self.calls[-1]["json"]

    @property
    def user_prompt(self):
        return self.payload["messages"][-1]["content"]
