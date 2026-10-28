"""
Text embeddings for retrieval and duplicate detection.

EMBEDDING_BACKEND=auto (default) -> fastembed if installed and its model loads, else hashing
EMBEDDING_BACKEND=fastembed      -> require fastembed (raises if unavailable)
EMBEDDING_BACKEND=hashing        -> dependency-free fallback, no model download
EMBEDDING_MODEL                  -> fastembed model (default BAAI/bge-small-en-v1.5)

The hashing embedder is a bag of word 1-2-grams hashed into a fixed-size vector. It is NOT
a semantic model: it matches shared wording, not meaning. It exists so the app, tests and CI
run anywhere. `get_embedder().name` (shown in /api/health) says which backend is active.
"""

import logging
import os
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

log = logging.getLogger(__name__)
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.where(norms == 0, 1, norms)).astype("float32")


class HashingEmbedder:
    kind = "hashing"

    def __init__(self, dim: int = 2048):
        self.dim = dim
        self.name = f"hashing-{dim}"
        self._vec = HashingVectorizer(n_features=dim, ngram_range=(1, 2), alternate_sign=False,
                                      norm=None, stop_words="english", lowercase=True)

    def embed(self, texts: List[str]) -> np.ndarray:
        return _normalize(self._vec.transform(texts).toarray().astype("float32"))


class FastEmbedEmbedder:
    kind = "fastembed"

    def __init__(self, model: str = DEFAULT_MODEL):
        from fastembed import TextEmbedding  # optional dependency
        self._model = TextEmbedding(model_name=model)
        self.name = f"fastembed:{model}"
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed(self, texts: List[str]) -> np.ndarray:
        return _normalize(np.array(list(self._model.embed(texts)), dtype="float32"))


_cached = None


def get_embedder():
    global _cached
    if _cached is not None:
        return _cached
    backend = os.environ.get("EMBEDDING_BACKEND", "auto").lower()
    if backend in ("auto", "fastembed"):
        try:
            _cached = FastEmbedEmbedder(os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL))
            return _cached
        except Exception as e:  # ImportError, model download failure, ONNX runtime problems...
            if backend == "fastembed":
                raise
            log.warning("fastembed unavailable (%s); falling back to the hashing embedder", e)
    elif backend != "hashing":
        raise ValueError(f"Unknown EMBEDDING_BACKEND '{backend}'")
    _cached = HashingEmbedder()
    return _cached


def reset_embedder(embedder: Optional[object] = None) -> None:
    global _cached
    _cached = embedder
