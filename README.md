# Support Triage Agent with MCP

![CI](https://github.com/vishwavardhan33-arch/Support-Triage-Agent-with-MCP/actions/workflows/ci.yml/badge.svg)

An AI triage desk for a home-loan partner-support team. It classifies each ticket by category, priority and
sentiment, drafts a first reply, flags duplicates, and answers questions about the queue. The same tools are exposed
two ways: as an **MCP server** (any MCP client) and as a **web dashboard** (FastAPI + plain HTML/CSS/JS).

## How it works

```
 MCP client ──┐                                    ┌─ embeddings ─ vector store (FAISS or NumPy)
              ├─► server.py (tools) ─► retrieval ──┤   similar resolved tickets, duplicate check
 Browser ─► app.py (REST) ─┘              │        └─ tickets.json
                                          └─► triage.py ─► llm.py ─► Groq (openai/gpt-oss-20b) or Ollama
```

| Piece | What it does |
|---|---|
| `server.py` | MCP tools: `list_tickets`, `get_ticket`, `update_ticket_status`, `create_ticket`, `find_duplicates`, `classify_ticket`, `ask_tickets` |
| `app.py` | REST API over the same functions, serves `static/`, rate limiting, health endpoint |
| `triage.py` | Prompt building, JSON parsing, label validation. One code path for the app **and** the evaluation |
| `llm.py` | Only place that talks to a model. Groq (default) or Ollama, with retry on 429 |
| `embeddings.py` | `fastembed` (BAAI/bge-small-en-v1.5) if installed, else a dependency-free hashing fallback |
| `vectorstore.py` | FAISS exact search if installed, else NumPy. Thread-safe (see Notes) |
| `retrieval.py` | `TicketIndex`: similar-ticket retrieval and duplicate detection |
| `evaluate.py` | Routing accuracy with confidence intervals, kNN baseline, LLM-as-judge, duplicate-detection sweep |
| `static/` | Dashboard: ledger table, case file, **New case** form, duplicate notice, ask box |

**Classification** retrieves the 3 most similar *resolved, labeled* tickets and shows them to the model as reference
examples. **Duplicate detection** flags a ticket when it shares a loan/partner ID with related wording, comes from the
same named sender with related wording, or is a near-copy from an anonymous sender. Two named people with different
IDs are never duplicates, however similar the text.

## Run it

```bash
pip install -r requirements.txt            # core (hashing embeddings, NumPy store)
pip install -r requirements-ml.txt         # optional: real embeddings (fastembed) + FAISS

export GROQ_API_KEY=...                    # Windows PowerShell: $env:GROQ_API_KEY="..."
python app.py                              # dashboard on http://localhost:8000
python server.py                           # MCP server over stdio (e.g. for Claude Desktop)
```

`GET /api/health` (and the header plate in the UI) reports which LLM, embedding model and vector store are active, so
you can tell whether you are on real embeddings or the fallback.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | none | required for the Groq provider |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | model for triage and Q&A |
| `LLM_PROVIDER` | `groq` | `ollama` for a local model (`OLLAMA_URL`, `OLLAMA_MODEL`) |
| `EMBEDDING_BACKEND` | `auto` | `fastembed`, `hashing` or `auto` (fastembed if available) |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model |
| `VECTOR_STORE` | `auto` | `numpy` to skip FAISS |
| `DUPLICATE_THRESHOLD` | per embedder | see Calibrating duplicates |
| `RATE_LIMIT_PER_MIN` / `RATE_LIMIT_GLOBAL_PER_MIN` | `10` / `60` | per client and overall, on endpoints that write or call the LLM |
| `MAX_TICKETS` | `600` | cap on queue size (public demo protection) |
| `TICKETS_PATH` | `data/tickets.json` | point at a scratch copy so local use does not dirty the repo |

## Evaluation

```bash
python evaluate.py classify --dataset synthetic            # 240 labeled home-loan tickets
python evaluate.py classify --dataset public               # 200 real bank queries
python evaluate.py classify --dataset public --mode zero-shot
python evaluate.py classify --dataset public --mode baseline   # no LLM: nearest-neighbour vote, free
python evaluate.py classify --dataset synthetic --judge --judge-model openai/gpt-oss-120b
python evaluate.py duplicates
```

Every report gives n, accuracy with a 95% Wilson interval, macro-F1, weakest categories and top confusions. LLM
failures (rate limits, outages) are excluded and reported separately, so an outage cannot change the score.
Unparseable model output counts as wrong.

**Datasets**

* `synthetic` (`data/generate_data.py`): 240 template-generated tickets. Subjects never contain the category name,
  about 15% are noisy (lower-cased, typos), about 10% mention a second problem, priorities depend on urgency cues, and
  24 are labeled duplicates plus 16 same-loan/different-issue hard negatives. It is still template-based, so it is the
  easier of the two sets.
* `public` (`data/fetch_public_dataset.py`): 200 held-out test queries from
  [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) (PolyAI, CC BY 4.0; Casanueva et al. 2020), with
  a disjoint 616-query retrieval pool. Its 77 intents are grouped into 9 routing queues **by me**, so quote results as
  "Banking77 grouped into 9 queues", not as a Banking77 benchmark. It has no priority labels.

**Reference numbers without an LLM** (kNN baseline, hashing embedder; reproduced in CI on every push):

| Set | n | Category accuracy (95% CI) |
|---|---|---|
| synthetic | 240 | 94.6% (91.0 to 96.8) |
| public | 200 | 69.5% (62.8 to 75.5) |

On the synthetic set retrieval alone is already at 94.6%, so a high LLM score there says little. Use `public`, and
compare `rag` against `zero-shot` and `baseline` to see what the model and the retrieval each contribute.

**LLM results** (fill in after running; do not quote numbers you have not measured):

| Set | Mode | Model | n | Category accuracy (95% CI) |
|---|---|---|---|---|
| public | rag | openai/gpt-oss-20b | 200 | _run it_ |
| public | zero-shot | openai/gpt-oss-20b | 200 | _run it_ |
| synthetic | rag | openai/gpt-oss-20b | 240 | _run it_ |

**Calibrating duplicates.** Similarity scores differ a lot between embedders, so the threshold must be tuned for the
one you use. The hashing fallback was measured: threshold 0.25 gives recall 1.00 and precision 0.55 (it cannot tell
"same loan, same issue, reworded" from "same loan, different issue"). The fastembed default of 0.88 is an
**uncalibrated guess**: run `python evaluate.py duplicates` with `requirements-ml.txt` installed and set
`DUPLICATE_THRESHOLD` to the value it recommends.

## Deploy on Render

Build: `pip install -r requirements.txt` (add `-r requirements-ml.txt` for real embeddings).
Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`. Set `GROQ_API_KEY` as a secret.

* **Memory.** The free tier has 512 MB. fastembed plus ONNX runtime may not fit; if the service restarts or is killed,
  set `EMBEDDING_BACKEND=hashing`. Check `/api/health` after deploy to see which backend loaded.
* **Storage is ephemeral.** Cases filed in the UI and status changes live in `data/tickets.json` and reset on restart
  or redeploy. A persistent disk or a database is the next step if that matters.
* **First start** downloads the embedding model (about 130 MB) before the service becomes ready.

## Tests and CI

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest          # 119 tests, no API key or model download needed
ruff check .
```

GitHub Actions runs lint and tests on Python 3.11 and 3.12 (with FAISS), then the LLM-free evaluations. Tests mock the
LLM, so they check the code, not the model; model quality is what `evaluate.py` measures. Dataset tests also guard
against label leakage in ticket subjects, and floor tests fail if retrieval quality drops.

## Notes and limitations

* A FAISS index crashes the process if one thread adds while another searches, and the web server uses a thread pool.
  `VectorStore` takes a lock on every operation, and `tests/test_index.py` has a regression test for it.
* Seeded tickets keep their ground-truth labels: "Run AI triage" on them only shows a prediction. Tickets filed in
  the UI are the ones whose category and priority get set.
* Ticket text is untrusted: it is HTML-escaped in the UI and the prompts tell the model to ignore instructions inside
  it. This reduces, but does not eliminate, prompt-injection risk.
* Groq is a third-party API, so ticket text leaves your infrastructure. For data that must stay on-prem, use
  `LLM_PROVIDER=ollama`.
* The in-memory rate limiter is per process, which is fine for one instance only.

## License

MIT. Banking77 data is CC BY 4.0 (see above).
