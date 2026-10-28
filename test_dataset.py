import importlib.util
import json
from collections import Counter
from pathlib import Path

import triage

ROOT = Path(__file__).parent.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "data" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed():
    return json.loads((ROOT / "data" / "tickets.json").read_text())


def test_seed_file_is_pristine():
    """Tickets filed through the UI belong in the live store, never in the committed seed."""
    polluted = [t["id"] for t in seed() if t.get("source") == "ui"]
    message = f"data/tickets.json contains UI-created tickets {polluted}; restore it with: python data/generate_data.py"
    assert polluted == [], message


def test_seed_set_is_large_and_well_formed():
    data = seed()
    assert len(data) >= 200
    assert len({t["id"] for t in data}) == len(data)
    assert all(t["category"] in triage.HOME_LOAN_CATEGORIES and t["priority"] in {"low", "medium", "high"} for t in data)
    assert all(t["status"] in {"open", "in_progress", "closed"} for t in data)


def test_every_category_has_enough_examples_in_both_pool_and_eval():
    counts = Counter(t["category"] for t in seed())
    assert set(counts) == set(triage.HOME_LOAN_CATEGORIES) and min(counts.values()) >= 15
    resolved = Counter(t["category"] for t in seed() if t["status"] != "open")
    assert min(resolved.values()) >= 5


def test_subjects_never_leak_the_label():
    """An earlier dataset put the category in the subject, so any model could read the answer."""
    names = [c.lower() for c in triage.HOME_LOAN_CATEGORIES]
    leaks = [t["subject"] for t in seed() if any(n in t["subject"].lower() for n in names)]
    assert leaks == []


def test_labeled_duplicates_point_at_real_tickets():
    data = seed()
    ids = {t["id"] for t in data}
    dups = [t for t in data if t.get("duplicate_of")]
    assert len(dups) == 24 and all(t["duplicate_of"] in ids for t in dups)
    by_id = {t["id"]: t for t in data}
    assert all(by_id[t["duplicate_of"]]["category"] == t["category"] for t in dups)


def test_dataset_has_hard_cases():
    data = seed()
    assert sum(bool(t.get("hard")) for t in data) >= 10
    assert sum(1 for t in data if t["subject"] == t["subject"].lower() and t["subject"]) >= 10  # noisy lower-cased tickets


def test_generator_and_server_agree_on_categories():
    assert list(load("generate_data").SPEC) == triage.HOME_LOAN_CATEGORIES


def test_public_sample():
    path = ROOT / "data" / "public" / "banking77_sample.json"
    d = json.loads(path.read_text())
    assert len(d["eval"]) >= 100 and len(d["pool"]) >= 300 and len(d["categories"]) == 9
    assert {x["body"].strip().lower() for x in d["eval"]}.isdisjoint({x["body"].strip().lower() for x in d["pool"]})
    assert all(x["category"] in d["categories"] for x in d["eval"] + d["pool"])
    assert d["license"] == "CC BY 4.0" and "Banking77" in d["source"]


def test_intent_grouping_covers_all_77_intents_once():
    mod = load("fetch_public_dataset")
    all_intents = [i for intents in mod.GROUPS.values() for i in intents]
    assert len(all_intents) == len(set(all_intents)) == 77
