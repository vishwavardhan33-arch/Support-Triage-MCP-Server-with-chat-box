import os
import shutil
from pathlib import Path

import pytest

# Must be set before the app modules are imported: tests never hit a real model or download one.
os.environ["EMBEDDING_BACKEND"] = "hashing"
os.environ["LLM_PROVIDER"] = "groq"
os.environ["GROQ_API_KEY"] = "test-key"
os.environ.pop("DUPLICATE_THRESHOLD", None)

import embeddings  # noqa: E402
import retrieval  # noqa: E402
import server  # noqa: E402

SEED_FILE = Path(server.__file__).parent / "data" / "tickets.json"


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Every test gets its own copy of tickets.json, a fresh index, and a fresh rate limiter."""
    data = tmp_path / "tickets.json"
    shutil.copy(SEED_FILE, data)
    monkeypatch.setattr(server, "DATA_PATH", data)
    embeddings.reset_embedder()
    retrieval.reset_index()
    import app as app_module
    app_module.per_ip.hits.clear()
    app_module.global_limit.hits.clear()
    yield
    retrieval.reset_index()
    embeddings.reset_embedder()


def call(tool, *args, **kwargs):
    """Invoke an @mcp.tool() function directly, whether or not the decorator wraps it."""
    return getattr(tool, "fn", tool)(*args, **kwargs)
