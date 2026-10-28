"""
LLM client: the only module that talks to a model.

LLM_PROVIDER=groq   (default) -> Groq's OpenAI-compatible chat API, needs GROQ_API_KEY
LLM_PROVIDER=ollama           -> a local Ollama server (no key)

Env: GROQ_API_KEY, GROQ_MODEL (default openai/gpt-oss-20b), OLLAMA_URL, OLLAMA_MODEL,
     LLM_MAX_TOKENS (default 1024), GROQ_REASONING_EFFORT (optional: low|medium|high)

If you already have a working Groq helper in your own server.py, you can keep it:
make `call_llm` below delegate to it. Everything else in the project calls only
`call_llm(system, user, json_mode=...)`.
"""

import json
import os
import re
import time
from typing import Optional

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_RETRIES = 3


class LLMError(RuntimeError):
    """Raised for any failure to get a usable response from the model."""


def provider() -> str:
    return os.environ.get("LLM_PROVIDER", "groq").lower()


def model_name() -> str:
    if provider() == "ollama":
        return os.environ.get("OLLAMA_MODEL", "llama3.2")
    return os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")


def llm_info() -> dict:
    """Cheap, no-network description of the configured LLM (used by /api/health)."""
    p = provider()
    configured = bool(os.environ.get("GROQ_API_KEY")) if p == "groq" else True
    return {"provider": p, "model": model_name(), "configured": configured}


def call_llm(system: str, user: str, json_mode: bool = True, model: Optional[str] = None,
             timeout: int = 90) -> str:
    p = provider()
    if p == "groq":
        return _call_groq(system, user, json_mode, model, timeout)
    if p == "ollama":
        return _call_ollama(system, user, json_mode, model, timeout)
    raise LLMError(f"Unknown LLM_PROVIDER '{p}'. Use 'groq' or 'ollama'.")


def _retry_wait(resp: requests.Response, attempt: int) -> float:
    try:
        return min(float(resp.headers.get("retry-after", "")), 30.0)
    except ValueError:
        return float(min(2 ** attempt, 20))


def _call_groq(system: str, user: str, json_mode: bool, model: Optional[str], timeout: int) -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY is not set. Create a key at console.groq.com and set it as an "
                       "environment variable (or set LLM_PROVIDER=ollama for a local model).")
    payload = {
        "model": model or model_name(),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_completion_tokens": int(os.environ.get("LLM_MAX_TOKENS", "1024")),
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if os.environ.get("GROQ_REASONING_EFFORT"):
        payload["reasoning_effort"] = os.environ["GROQ_REASONING_EFFORT"]
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    attempt = 0
    while True:
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=timeout)
        except requests.exceptions.RequestException as e:
            raise LLMError(f"Could not reach Groq: {e}") from e

        if resp.status_code == 429 and attempt < MAX_RETRIES:
            time.sleep(_retry_wait(resp, attempt))
            attempt += 1
            continue
        if resp.status_code == 400 and "response_format" in payload and "response_format" in resp.text:
            payload.pop("response_format")  # model/endpoint without JSON mode: rely on the prompt
            continue
        if resp.status_code >= 400:
            raise LLMError(f"Groq returned HTTP {resp.status_code}: {resp.text[:200]}")
        break

    try:
        return resp.json()["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise LLMError("Unexpected response shape from Groq") from e


def _call_ollama(system: str, user: str, json_mode: bool, model: Optional[str], timeout: int) -> str:
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    payload = {"model": model or model_name(), "system": system, "prompt": user, "stream": False}
    if json_mode:
        payload["format"] = "json"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["response"]
    except requests.exceptions.RequestException as e:
        raise LLMError(f"Could not reach Ollama at {url}. Is `ollama serve` running and the model pulled? ({e})") from e
    except (KeyError, ValueError) as e:
        raise LLMError("Unexpected response shape from Ollama") from e


def parse_json_object(text: str) -> dict:
    """Parse a JSON object out of model output, tolerating code fences and stray prose."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in model output") from None
        value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("model output was JSON but not an object")
    return value
