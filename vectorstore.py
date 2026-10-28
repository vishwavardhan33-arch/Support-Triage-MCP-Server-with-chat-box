"""
Tiny vector store over L2-normalised vectors (inner product == cosine similarity).

Uses FAISS (IndexFlatIP, exact search) when `faiss-cpu` is installed, otherwise a NumPy
matrix multiply. Both are exact, so results are identical; FAISS just scales further.
VECTOR_STORE=numpy forces the fallback.

Thread safety: a FAISS index must not be added to while another thread searches it (that
segfaults the process), and the web server handles requests on a thread pool. Every public
operation therefore takes one lock.
"""

import os
import threading
from typing import Callable, List, Optional, Tuple

import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.ids: List[str] = []
        self._pos = {}
        self._lock = threading.RLock()
        self._faiss = None
        self._matrix = np.zeros((0, dim), dtype="float32")
        self.backend = "numpy"
        if os.environ.get("VECTOR_STORE", "auto").lower() != "numpy":
            try:
                import faiss
                self._faiss = faiss.IndexFlatIP(dim)
                self.backend = "faiss"
            except ImportError:
                pass

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._pos

    def add(self, ids: List[str], vectors: np.ndarray) -> None:
        with self._lock:
            keep = [i for i, item_id in enumerate(ids) if item_id not in self._pos]
            if not keep:
                return
            vectors = np.ascontiguousarray(vectors[keep], dtype="float32")
            for i in keep:
                self._pos[ids[i]] = len(self.ids)
                self.ids.append(ids[i])
            if self._faiss is not None:
                self._faiss.add(vectors)
            else:
                self._matrix = np.vstack([self._matrix, vectors])

    def vector(self, item_id: str) -> np.ndarray:
        with self._lock:
            pos = self._pos[item_id]
            if self._faiss is not None:
                return self._faiss.reconstruct(pos)
            return self._matrix[pos].copy()

    def search(self, query: np.ndarray, k: int = 5,
               predicate: Optional[Callable[[str], bool]] = None) -> List[Tuple[str, float]]:
        """Top-k (id, similarity), best first. With a predicate, filters before taking top-k."""
        q = np.ascontiguousarray(query.reshape(1, -1), dtype="float32")
        with self._lock:
            if not self.ids:
                return []
            fetch = len(self.ids) if predicate else min(k, len(self.ids))
            if self._faiss is not None:
                scores, idx = self._faiss.search(q, fetch)
                pairs = [(self.ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i != -1]
            else:
                scores = self._matrix @ q[0]
                order = np.argsort(-scores)[:fetch]
                pairs = [(self.ids[i], float(scores[i])) for i in order]
        if predicate:
            pairs = [p for p in pairs if predicate(p[0])]
        return pairs[:k]
