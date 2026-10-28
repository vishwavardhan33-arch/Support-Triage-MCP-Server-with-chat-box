import numpy as np
import pytest

import embeddings
from embeddings import HashingEmbedder
from vectorstore import VectorStore


def test_hashing_embeddings_are_normalised_and_deterministic():
    e = HashingEmbedder()
    a = e.embed(["funds not received for BHL-12345", "cannot log in"])
    assert a.shape == (2, e.dim)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.allclose(a, e.embed(["funds not received for BHL-12345", "cannot log in"]))


def test_similar_wording_scores_higher_than_unrelated():
    e = HashingEmbedder()
    v = e.embed(["money not credited for BHL-12345", "funds not credited for BHL-12345", "partner cannot log in to dashboard"])
    assert v[0] @ v[1] > 0.4 > v[0] @ v[2]


def test_empty_text_does_not_produce_nan():
    assert not np.isnan(HashingEmbedder().embed(["", "   "])).any()


def test_backend_selection(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "hashing")
    embeddings.reset_embedder()
    assert isinstance(embeddings.get_embedder(), HashingEmbedder)
    monkeypatch.setenv("EMBEDDING_BACKEND", "bogus")
    embeddings.reset_embedder()
    with pytest.raises(ValueError):
        embeddings.get_embedder()


def test_auto_falls_back_when_fastembed_unavailable(monkeypatch):
    def boom(*a, **k):
        raise ImportError("no fastembed")
    monkeypatch.setattr(embeddings, "FastEmbedEmbedder", boom)
    monkeypatch.setenv("EMBEDDING_BACKEND", "auto")
    embeddings.reset_embedder()
    assert isinstance(embeddings.get_embedder(), HashingEmbedder)
    monkeypatch.setenv("EMBEDDING_BACKEND", "fastembed")  # explicit request must not silently degrade
    embeddings.reset_embedder()
    with pytest.raises(ImportError):
        embeddings.get_embedder()


def _vectors(n=6, dim=16, seed=0):
    m = np.random.default_rng(seed).normal(size=(n, dim)).astype("float32")
    return m / np.linalg.norm(m, axis=1, keepdims=True)


@pytest.fixture(params=["numpy", "faiss"])
def store(request, monkeypatch):
    if request.param == "faiss":
        pytest.importorskip("faiss")
    monkeypatch.setenv("VECTOR_STORE", request.param)
    s = VectorStore(16)
    assert s.backend == request.param
    return s


def test_search_returns_best_first(store):
    vecs = _vectors()
    store.add([f"id{i}" for i in range(6)], vecs)
    hits = store.search(vecs[2], k=3)
    assert hits[0][0] == "id2" and hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert [s for _, s in hits] == sorted([s for _, s in hits], reverse=True)


def test_predicate_filters_before_top_k(store):
    vecs = _vectors()
    store.add([f"id{i}" for i in range(6)], vecs)
    hits = store.search(vecs[2], k=2, predicate=lambda i: i != "id2")
    assert len(hits) == 2 and all(i != "id2" for i, _ in hits)


def test_duplicate_ids_are_ignored_and_empty_store_is_safe(store):
    assert store.search(_vectors()[0], k=3) == []
    vecs = _vectors()
    store.add(["a", "b"], vecs[:2])
    store.add(["b", "c"], vecs[1:3])
    assert len(store) == 3 and "c" in store
    assert np.allclose(store.vector("c"), vecs[2], atol=1e-5)


def test_faiss_and_numpy_agree(monkeypatch):
    pytest.importorskip("faiss")
    vecs = _vectors(n=40, dim=16, seed=3)
    results = {}
    for backend in ("numpy", "faiss"):
        monkeypatch.setenv("VECTOR_STORE", backend)
        s = VectorStore(16)
        s.add([str(i) for i in range(40)], vecs)
        results[backend] = s.search(vecs[7], k=5)
    assert [i for i, _ in results["numpy"]] == [i for i, _ in results["faiss"]]
