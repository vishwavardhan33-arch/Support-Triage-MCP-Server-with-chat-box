"""
Evaluation harness.

  python evaluate.py classify --dataset synthetic            # 240 labeled home-loan tickets
  python evaluate.py classify --dataset public               # 200 real bank queries (Banking77, 9 queues)
  python evaluate.py classify --dataset public --mode zero-shot
  python evaluate.py classify --dataset public --mode baseline   # embedding kNN, no LLM, free and offline
  python evaluate.py classify --dataset synthetic --judge --judge-model openai/gpt-oss-120b
  python evaluate.py duplicates                              # precision/recall sweep for duplicate detection

Modes
  rag        LLM + top-k similar labeled tickets from the retrieval pool (the production path)
  zero-shot  LLM only, no retrieved examples
  baseline   no LLM: majority vote of the nearest labeled tickets (what retrieval alone achieves)

Reading the numbers
  * Always quote n and the 95% interval. On a few hundred items the interval is several points wide.
  * Compare rag against zero-shot and baseline. If baseline is already near rag, the retrieval pool is doing the work.
  * synthetic is template-generated, so it is the easier of the two sets. public is real text, but its 9 queues
    are my own grouping of Banking77's 77 intents: quote it as "Banking77 grouped into 9 queues".
  * Priority labels exist only for synthetic; public reports category accuracy only.
  * Tickets whose LLM call failed (rate limit, outage) are excluded and reported separately, so an outage
    cannot silently lower or raise the score. Unparseable model output counts as wrong.
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

import llm
import triage
from retrieval import TicketIndex, duplicate_verdict, extract_ids, ticket_text

ROOT = Path(__file__).parent
SEED = 11


# ------------------------------------------------------------------ metrics
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(records: Sequence[dict], labels: Sequence[str], true_key: str = "true", pred_key: str = "pred") -> dict:
    """Accuracy (+95% CI), macro-F1, per-label recall and the most common confusions."""
    scored = [r for r in records if not r.get("error")]
    n = len(scored)
    correct = sum(1 for r in scored if r[pred_key] == r[true_key])
    f1s, per_label = [], {}
    for label in labels:
        tp = sum(1 for r in scored if r[true_key] == label and r[pred_key] == label)
        support = sum(1 for r in scored if r[true_key] == label)
        predicted = sum(1 for r in scored if r[pred_key] == label)
        recall = tp / support if support else None
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if recall and (precision + recall) else 0.0
        if support:
            f1s.append(f1)
        per_label[label] = {"support": support, "recall": None if recall is None else round(recall, 3)}
    lo, hi = wilson_interval(correct, n)
    confusions = Counter((r[true_key], r[pred_key]) for r in scored if r[pred_key] != r[true_key])
    return {
        "n": n, "n_errors": len(records) - n, "accuracy": round(correct / n, 4) if n else None,
        "ci95": [round(lo, 4), round(hi, 4)], "macro_f1": round(statistics.mean(f1s), 4) if f1s else None,
        "per_label": per_label,
        "top_confusions": [{"true": t, "pred": p, "count": c} for (t, p), c in confusions.most_common(5)],
    }


# ------------------------------------------------------------------ datasets
def dup_groups(tickets: Sequence[dict]) -> Dict[str, set]:
    """id -> ids of the same duplicate group (the original and its duplicates)."""
    parent = {t["id"]: t["id"] for t in tickets}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for t in tickets:
        if t.get("duplicate_of") in parent:
            parent[find(t["id"])] = find(t["duplicate_of"])
    members = defaultdict(set)
    for t in tickets:
        members[find(t["id"])].add(t["id"])
    return {i: members[find(i)] for i in parent}


def load_synthetic() -> dict:
    tickets = json.loads((ROOT / "data" / "tickets.json").read_text(encoding="utf-8"))
    labeled = [t for t in tickets if t.get("source") != "ui" and t.get("category") in triage.HOME_LOAN_CATEGORIES]
    return {"name": "synthetic", "eval": labeled,
            # only resolved tickets are trusted as examples, mirroring production
            "pool": [t for t in labeled if t["status"] != "open"],
            "categories": triage.HOME_LOAN_CATEGORIES, "domain": triage.HOME_LOAN_DOMAIN, "has_priority": True,
            "groups": dup_groups(labeled)}


def load_public() -> dict:
    path = ROOT / "data" / "public" / "banking77_sample.json"
    if not path.exists():
        raise SystemExit("Missing data/public/banking77_sample.json. Run: python data/fetch_public_dataset.py")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"name": "public", "eval": data["eval"], "pool": data["pool"], "categories": data["categories"],
            "domain": "a digital bank's customer-support desk", "has_priority": False, "groups": {}}


# ------------------------------------------------------------------ classification
def knn_vote(neighbors: Sequence[tuple]) -> Optional[str]:
    votes: Dict[str, float] = defaultdict(float)
    for ticket, score in neighbors:
        votes[ticket["category"]] += max(score, 0.0)
    return max(votes, key=votes.get) if votes else None


def run_classification(ds: dict, mode: str, embedder, n: Optional[int] = None, k: int = 3,
                       model: Optional[str] = None, sleep: float = 0.0, judge_model: Optional[str] = None,
                       classify_fn: Callable = triage.classify, judge_fn: Optional[Callable] = None,
                       progress: Callable = print) -> List[dict]:
    items = list(ds["eval"])
    if n and n < len(items):
        items = random.Random(SEED).sample(items, n)
    index = TicketIndex(embedder)
    index.sync(ds["pool"])
    records = []
    for i, t in enumerate(items, 1):
        exclude = ds["groups"].get(t["id"], set()) - {t["id"]}  # a ticket's own duplicates must not leak its label
        neighbors = index.similar(t, ds["pool"], k=5 if mode == "baseline" else k, exclude_ids=exclude)
        rec = {"id": t["id"], "true": t["category"], "true_priority": t.get("priority") if ds["has_priority"] else None,
               "hard": bool(t.get("hard"))}
        if mode == "baseline":
            rec["pred"] = knn_vote(neighbors)
        else:
            examples = [] if mode == "zero-shot" else [x for x, _ in neighbors]
            try:
                out = classify_fn(t, examples, ds["categories"], ds["domain"], model=model)
                rec.update(pred=out["category"] if out["category_valid"] else None, pred_priority=out["priority"],
                           response=out["suggested_response"])
            except triage.TriageParseError as e:
                rec.update(pred=None, parse_failed=True, raw=e.raw[:200])
            except llm.LLMError as e:
                rec.update(pred=None, error=str(e)[:200])
            if sleep:
                time.sleep(sleep)
            if judge_model and rec.get("response") and not rec.get("error"):
                rec["judge"] = (judge_fn or judge_response)(t, rec["response"], judge_model)
        records.append(rec)
        if i % 10 == 0 or i == len(items):
            progress(f"  {i}/{len(items)}")
    return records


JUDGE_SYSTEM = (
    "You grade an AI-drafted first reply to a support ticket. Be critical; do not default to high scores. "
    'Respond with ONLY JSON: {"relevance": 1-5, "professionalism": 1-5, "actionability": 1-5, "overall": 1-5}. '
    "relevance: addresses what the ticket is actually about. professionalism: tone and clarity. "
    "actionability: tells the sender what happens next without inventing facts."
)


def judge_response(ticket: dict, response: str, judge_model: str) -> dict:
    prompt = f"TICKET: {ticket.get('subject', '')}. {ticket.get('body', '')}\n\nDRAFTED REPLY:\n{response}"
    try:
        return llm.parse_json_object(llm.call_llm(JUDGE_SYSTEM, prompt, json_mode=True, model=judge_model))
    except (llm.LLMError, ValueError) as e:
        return {"error": str(e)[:120]}


def summarize_judge(records: Sequence[dict]) -> dict:
    judged = [r["judge"] for r in records if isinstance(r.get("judge"), dict) and "overall" in r["judge"]]
    out = {"n": len(judged)}
    for key in ("relevance", "professionalism", "actionability", "overall"):
        vals = [j[key] for j in judged if isinstance(j.get(key), (int, float))]
        if vals:
            out[key] = round(statistics.mean(vals), 2)
    return out


def classify_report(ds: dict, mode: str, records: List[dict], meta: dict) -> dict:
    report = {"meta": meta, "category": summarize(records, ds["categories"])}
    if ds["has_priority"] and mode != "baseline":
        pr = [{**r, "true": r["true_priority"], "pred": r.get("pred_priority")} for r in records]
        report["priority"] = summarize(pr, ["low", "medium", "high"])
    hard = [r for r in records if r.get("hard")]
    if hard:
        report["hard_subset"] = summarize(hard, ds["categories"])
    judge = summarize_judge(records)
    if judge["n"]:
        report["judge"] = judge
    report["parse_failures"] = sum(1 for r in records if r.get("parse_failed"))
    return report


def print_report(report: dict) -> None:
    m, c = report["meta"], report["category"]
    print(f"\n=== {m['dataset']} / {m['mode']} / {m['model']} / embeddings {m['embeddings']} ===")
    lo, hi = c["ci95"]
    print(f"Category accuracy: {c['accuracy']:.1%} (95% CI {lo:.1%} to {hi:.1%}), n={c['n']}, macro-F1 {c['macro_f1']:.3f}")
    if c["n_errors"]:
        print(f"WARNING: {c['n_errors']} ticket(s) failed with an LLM error and were excluded. Re-run before quoting numbers.")
    if report.get("parse_failures"):
        print(f"{report['parse_failures']} unparseable model output(s) counted as wrong.")
    if "priority" in report:
        p = report["priority"]
        print(f"Priority accuracy: {p['accuracy']:.1%} (95% CI {p['ci95'][0]:.1%} to {p['ci95'][1]:.1%})")
    if "hard_subset" in report:
        h = report["hard_subset"]
        print(f"Mixed-issue subset: {h['accuracy']:.1%}, n={h['n']}")
    if "judge" in report:
        print("LLM-as-judge (1-5): " + ", ".join(f"{k} {v}" for k, v in report["judge"].items() if k != "n")
              + f" (n={report['judge']['n']})")
    print("Most common confusions:")
    for x in c["top_confusions"]:
        print(f"  {x['true']} -> {x['pred']}: {x['count']}")
    weak = sorted((v["recall"], k) for k, v in c["per_label"].items() if v["recall"] is not None)[:3]
    print("Weakest categories by recall: " + ", ".join(f"{k} ({r:.0%})" for r, k in weak))


# ------------------------------------------------------------------ duplicate detection
def eval_duplicates(tickets: Sequence[dict], embedder, thresholds: Optional[Sequence[float]] = None) -> dict:
    """Pairwise precision/recall of duplicate_verdict against the labeled duplicate_of pairs."""
    tickets = [t for t in tickets if t.get("source") != "ui"]
    texts = [ticket_text(t) for t in tickets]
    vectors = embedder.embed(texts)
    sims = vectors @ vectors.T
    ids = [extract_ids(x) for x in texts]
    senders = [str(t.get("sender", "")) for t in tickets]
    pos = {t["id"]: i for i, t in enumerate(tickets)}
    truth = {tuple(sorted((pos[t["id"]], pos[t["duplicate_of"]]))): ("id" if t.get("reference_id") else "followup")
             for t in tickets if t.get("duplicate_of") in pos}
    rows = []
    for thr in (thresholds if thresholds is not None else np.round(np.arange(0.10, 0.961, 0.01), 2)):
        predicted = set()
        for i in range(len(tickets)):
            for j in range(i + 1, len(tickets)):
                if duplicate_verdict(float(sims[i, j]), ids[i], ids[j], senders[i], senders[j], float(thr)):
                    predicted.add((i, j))
        tp = len(predicted & truth.keys())
        precision = tp / len(predicted) if predicted else 1.0
        recall = tp / len(truth) if truth else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        by_kind = {kind: round(sum(1 for p in predicted & truth.keys() if truth[p] == kind)
                               / max(1, sum(1 for v in truth.values() if v == kind)), 3) for kind in ("id", "followup")}
        rows.append({"threshold": float(thr), "precision": round(precision, 3), "recall": round(recall, 3),
                     "f1": round(f1, 3), "false_positives": len(predicted) - tp, "recall_by_kind": by_kind})
    best = max(rows, key=lambda r: (r["f1"], r["threshold"]))  # ties: the most conservative threshold
    return {"n_tickets": len(tickets), "n_true_pairs": len(truth), "rows": rows, "best": best}


def print_duplicates(report: dict, embedder_name: str) -> None:
    print(f"\n=== duplicate detection / embeddings {embedder_name} ===")
    print(f"{report['n_tickets']} tickets, {report['n_true_pairs']} labeled duplicate pairs "
          f"(16 same-ID paraphrases, 8 same-sender follow-ups without an ID, plus 16 same-ID/different-issue hard negatives)")
    print("threshold  precision  recall   f1   false+  recall(id)  recall(follow-up)")
    for r in report["rows"]:
        if round(r["threshold"] * 100) % 5 == 0:
            kinds = r["recall_by_kind"]
            print(f"  {r['threshold']:.2f}      {r['precision']:.3f}    {r['recall']:.3f}  {r['f1']:.3f}   "
                  f"{r['false_positives']:>4}     {kinds['id']:.3f}       {kinds['followup']:.3f}")
    b = report["best"]
    print(f"Best F1 {b['f1']} at threshold {b['threshold']} (precision {b['precision']}, recall {b['recall']}). "
          f"Set DUPLICATE_THRESHOLD={b['threshold']} if you use this embedder.")


# ------------------------------------------------------------------ CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the support-triage system")
    sub = parser.add_subparsers(dest="task", required=True)
    c = sub.add_parser("classify", help="routing accuracy of the classifier")
    c.add_argument("--dataset", choices=["synthetic", "public"], default="synthetic")
    c.add_argument("--mode", choices=["rag", "zero-shot", "baseline"], default="rag")
    c.add_argument("--n", type=int, default=None, help="evaluate a random subset of this size (default: all)")
    c.add_argument("--k", type=int, default=3, help="retrieved examples per ticket in rag mode")
    c.add_argument("--model", default=None, help="override the classifier model")
    c.add_argument("--judge", action="store_true", help="also grade the drafted replies with an LLM judge")
    c.add_argument("--judge-model", default=None, help="a different model than the classifier reduces self-grading bias")
    c.add_argument("--sleep", type=float, default=0.0, help="seconds between LLM calls (rate limits)")
    sub.add_parser("duplicates", help="precision/recall sweep for duplicate detection")
    parser.add_argument("--embeddings", choices=["auto", "fastembed", "hashing"], help="override EMBEDDING_BACKEND")
    args = parser.parse_args(argv)

    if args.embeddings:
        os.environ["EMBEDDING_BACKEND"] = args.embeddings
    from embeddings import get_embedder
    embedder = get_embedder()
    out_dir = ROOT / "eval_results"
    out_dir.mkdir(exist_ok=True)

    if args.task == "duplicates":
        tickets = json.loads((ROOT / "data" / "tickets.json").read_text(encoding="utf-8"))
        report = eval_duplicates(tickets, embedder)
        print_duplicates(report, embedder.name)
        (out_dir / "duplicates.json").write_text(json.dumps({"embeddings": embedder.name, **report}, indent=1))
        return 0

    ds = load_public() if args.dataset == "public" else load_synthetic()
    if args.mode != "baseline" and not llm.llm_info()["configured"]:
        print("GROQ_API_KEY is not set (or use LLM_PROVIDER=ollama). Use --mode baseline for an LLM-free run.")
        return 2
    judge_model = (args.judge_model or llm.model_name()) if args.judge else None
    print(f"Evaluating {ds['name']} ({len(ds['eval'])} tickets, pool {len(ds['pool'])}) in {args.mode} mode...")
    records = run_classification(ds, args.mode, embedder, n=args.n, k=args.k, model=args.model, sleep=args.sleep,
                                 judge_model=judge_model)
    model = "none" if args.mode == "baseline" else (args.model or llm.model_name())
    meta = {"dataset": ds["name"], "mode": args.mode, "model": model,
            "judge_model": judge_model, "embeddings": embedder.name, "k": args.k, "seed": SEED,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
    report = classify_report(ds, args.mode, records, meta)
    print_report(report)
    (out_dir / f"{ds['name']}_{args.mode}.json").write_text(json.dumps({**report, "records": records}, indent=1))
    print(f"\nPer-ticket results saved to eval_results/{ds['name']}_{args.mode}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
