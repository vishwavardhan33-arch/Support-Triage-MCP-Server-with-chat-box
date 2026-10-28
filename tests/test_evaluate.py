import pytest

import evaluate
import llm
import triage
from embeddings import HashingEmbedder


def rec(true, pred, **kw):
    return {"true": true, "pred": pred, **kw}


def test_wilson_interval():
    assert evaluate.wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = evaluate.wilson_interval(50, 100)
    assert (lo, hi) == (pytest.approx(0.404, abs=0.002), pytest.approx(0.596, abs=0.002))
    lo, hi = evaluate.wilson_interval(100, 100)
    assert hi == pytest.approx(1.0) and lo < 1.0


def test_summarize_perfect_and_confusions():
    labels = ["a", "b"]
    s = evaluate.summarize([rec("a", "a"), rec("b", "b")], labels)
    assert s["accuracy"] == 1.0 and s["macro_f1"] == 1.0 and s["top_confusions"] == []
    s = evaluate.summarize([rec("a", "a"), rec("a", "b"), rec("a", "b"), rec("b", "b")], labels)
    assert s["accuracy"] == 0.5 and s["top_confusions"] == [{"true": "a", "pred": "b", "count": 2}]
    assert s["per_label"]["a"]["recall"] == pytest.approx(1 / 3, abs=0.001)


def test_summarize_excludes_llm_errors_but_counts_wrong_predictions():
    s = evaluate.summarize([rec("a", "a"), rec("a", None, error="rate limited"), rec("b", None)], ["a", "b"])
    assert s["n"] == 2 and s["n_errors"] == 1 and s["accuracy"] == 0.5


def test_dup_groups_union_original_and_copies():
    t = [{"id": "1"}, {"id": "2", "duplicate_of": "1"}, {"id": "3", "duplicate_of": "1"}, {"id": "4"}]
    g = evaluate.dup_groups(t)
    assert g["1"] == g["2"] == g["3"] == {"1", "2", "3"} and g["4"] == {"4"}


def test_knn_vote_is_similarity_weighted():
    nb = [({"category": "x"}, 0.9), ({"category": "y"}, 0.4), ({"category": "y"}, 0.4)]
    assert evaluate.knn_vote(nb) == "x" and evaluate.knn_vote([]) is None


def tiny_dataset():
    pool = [{"id": f"P{i}", "subject": "", "body": text, "category": cat, "status": "closed"}
            for i, (text, cat) in enumerate([("card has not arrived yet", "Cards"), ("my card never came in the post", "Cards"),
                                              ("transfer to my friend failed", "Transfers"),
                                              ("the transfer is still pending", "Transfers")])]
    items = [{"id": "E0", "subject": "", "body": "where is my card, not arrived", "category": "Cards", "status": "open"},
             {"id": "E1", "subject": "", "body": "transfer failed again", "category": "Transfers", "status": "open"}]
    return {"name": "tiny", "eval": items, "pool": pool, "categories": ["Cards", "Transfers"], "domain": "a bank",
            "has_priority": False, "groups": {}}


def fake_classify(seen):
    def fn(ticket, examples, categories, domain, model=None):
        seen.append((ticket["id"], [e["id"] for e in examples], list(categories)))
        return {"category": "Cards", "category_valid": True, "priority": "low", "sentiment": "neutral",
                "suggested_response": "Thanks.", "similar_ticket_ids": []}
    return fn


def test_rag_mode_passes_retrieved_examples_and_zero_shot_passes_none():
    seen = []
    ds, emb = tiny_dataset(), HashingEmbedder()
    out = evaluate.run_classification(ds, "rag", emb, k=2, classify_fn=fake_classify(seen), progress=lambda m: None)
    assert [r["pred"] for r in out] == ["Cards", "Cards"]
    assert seen[0][1] and set(seen[0][1]) <= {"P0", "P1", "P2", "P3"} and seen[0][2] == ["Cards", "Transfers"]
    seen.clear()
    evaluate.run_classification(ds, "zero-shot", emb, classify_fn=fake_classify(seen), progress=lambda m: None)
    assert all(examples == [] for _, examples, _ in seen)


def test_baseline_mode_makes_no_llm_calls():
    def boom(*a, **k):
        raise AssertionError("LLM must not be called in baseline mode")
    out = evaluate.run_classification(tiny_dataset(), "baseline", HashingEmbedder(), classify_fn=boom, progress=lambda m: None)
    assert [r["pred"] for r in out] == ["Cards", "Transfers"]


def test_leave_one_out_keeps_the_ticket_and_its_duplicates_out_of_its_examples():
    ds = tiny_dataset()
    ds["pool"].append({"id": "E0", **{k: v for k, v in ds["eval"][0].items() if k != "id"}, "status": "closed"})
    ds["groups"] = {"E0": {"E0", "DUP"}}
    seen = []
    evaluate.run_classification(ds, "rag", HashingEmbedder(), classify_fn=fake_classify(seen), progress=lambda m: None)
    assert "E0" not in seen[0][1]


def test_failures_are_recorded_not_raised():
    def flaky(ticket, examples, categories, domain, model=None):
        if ticket["id"] == "E0":
            raise llm.LLMError("429 again")
        raise triage.TriageParseError("bad", "garbage output")
    out = evaluate.run_classification(tiny_dataset(), "rag", HashingEmbedder(), classify_fn=flaky, progress=lambda m: None)
    assert out[0]["error"] and out[1]["parse_failed"] and out[1]["pred"] is None
    s = evaluate.summarize(out, ["Cards", "Transfers"])
    assert s["n"] == 1 and s["n_errors"] == 1 and s["accuracy"] == 0.0


def test_judge_scores_are_aggregated():
    judged = lambda t, r, m: {"relevance": 4, "professionalism": 5, "actionability": 3, "overall": 4}  # noqa: E731
    out = evaluate.run_classification(tiny_dataset(), "rag", HashingEmbedder(), classify_fn=fake_classify([]),
                                      judge_model="m", judge_fn=judged, progress=lambda m: None)
    assert evaluate.summarize_judge(out) == {"n": 2, "relevance": 4, "professionalism": 5, "actionability": 3, "overall": 4}


def test_judge_response_survives_bad_output(monkeypatch):
    monkeypatch.setattr(llm, "call_llm", lambda *a, **k: "not json")
    assert "error" in evaluate.judge_response({"subject": "s", "body": "b"}, "reply", "m")


def test_real_datasets_load_with_sane_pools():
    syn, pub = evaluate.load_synthetic(), evaluate.load_public()
    assert len(syn["eval"]) >= 200 and all(t["status"] != "open" for t in syn["pool"]) and syn["has_priority"]
    assert len(pub["eval"]) >= 100 and not pub["has_priority"] and len(pub["categories"]) == 9


def test_report_has_priority_only_when_labels_exist():
    syn = evaluate.load_synthetic()
    out = evaluate.run_classification(syn, "rag", HashingEmbedder(), n=12, classify_fn=fake_classify([]), progress=lambda m: None)
    meta = {"dataset": "synthetic", "mode": "rag", "model": "m", "embeddings": "e"}
    assert "priority" in evaluate.classify_report(syn, "rag", out, meta)
    assert "priority" not in evaluate.classify_report(syn, "baseline", out, meta)


def test_duplicate_sweep_on_the_seed_set():
    import json
    from pathlib import Path
    tickets = json.loads((Path(evaluate.ROOT) / "data" / "tickets.json").read_text())
    report = evaluate.eval_duplicates(tickets, HashingEmbedder(), thresholds=[0.25, 0.9])
    low, high = report["rows"]
    assert report["n_true_pairs"] == 24 and low["recall"] >= 0.9 and high["recall"] == 0.0
    assert report["best"]["threshold"] == 0.25


def test_cli_refuses_llm_modes_without_a_key(monkeypatch, capsys):
    monkeypatch.delenv("GROQ_API_KEY")
    assert evaluate.main(["--embeddings", "hashing", "classify", "--dataset", "public"]) == 2
    assert "baseline" in capsys.readouterr().out


def test_offline_retrieval_quality_floors():
    """
    Floors for the LLM-free kNN baseline with the hashing embedder, set below what was measured
    (public 69.5%, synthetic 94.6%). If retrieval or the datasets regress, this fails in CI without needing a model.
    """
    emb = HashingEmbedder()
    for loader, floor in ((evaluate.load_public, 0.62), (evaluate.load_synthetic, 0.90)):
        ds = loader()
        records = evaluate.run_classification(ds, "baseline", emb, progress=lambda m: None)
        assert evaluate.summarize(records, ds["categories"])["accuracy"] >= floor, ds["name"]


def test_duplicate_detection_f1_floor():
    import json
    from pathlib import Path
    tickets = json.loads((Path(evaluate.ROOT) / "data" / "tickets.json").read_text())
    row = evaluate.eval_duplicates(tickets, HashingEmbedder(), thresholds=[0.25])["rows"][0]
    assert row["recall"] >= 0.95 and row["f1"] >= 0.65  # measured: recall 1.0, F1 0.706 (precision 0.545)
