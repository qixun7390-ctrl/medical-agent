import json
from pathlib import Path
from typing import List

import faiss
import numpy as np

from app.core.config import settings

class VectorStore:
    def __init__(self):
        self.index_path = Path(settings.vector_index_path)
        self.meta_path = Path(settings.vector_meta_path)
        self.model = None
        self._model_loaded = False
        self.index = None
        self.meta = []
        self._tfidf = None
        self._tfidf_matrix = None
        self._load_meta()
        self._load_index()

    def _load_meta(self):
        if not self.meta_path.exists():
            self.meta = []
            return
        self.meta = []
        with self.meta_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    self.meta.append(json.loads(line))
                except Exception:
                    continue

    def _load_index(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))

    def _ensure_model(self):
        if self._model_loaded:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(settings.embedding_model)
            self._model_loaded = True
            return True
        except Exception:
            return False

    def _ensure_tfidf(self):
        if self._tfidf is not None and self._tfidf_matrix is not None:
            return
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except Exception:
            return
        texts = [m.get("snippet", "") for m in self.meta]
        self._tfidf = TfidfVectorizer(max_features=50000)
        self._tfidf_matrix = self._tfidf.fit_transform(texts)

    def _search_tfidf(self, query: str, top_k: int = 10):
        if self._tfidf is None or self._tfidf_matrix is None:
            return []
        q_vec = self._tfidf.transform([query])
        scores = (self._tfidf_matrix @ q_vec.T).toarray().ravel()
        if scores.size == 0:
            return []
        max_score = float(scores.max()) if scores.size else 0.0
        idxs = np.argsort(-scores)[:top_k]
        hits = []
        for idx in idxs:
            if idx < 0 or idx >= len(self.meta):
                continue
            m = self.meta[idx]
            sc = float(scores[idx])
            if max_score <= 0.0:
                sc = 0.01  # ensure passes min score filter when TF-IDF is flat
            hits.append({
                "doc_id": m.get("doc_id"),
                "score": sc,
                "snippet": m.get("snippet", ""),
                "source": m.get("source"),
                "retriever": "tfidf",
            })
        return hits

    def search(self, query: str, top_k: int = 10):
        if self.index is None or not self.meta:
            return []
        if self._ensure_model():
            q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
            scores, idxs = self.index.search(q_emb, top_k)
            hits = []
            for score, idx in zip(scores[0], idxs[0]):
                if idx < 0 or idx >= len(self.meta):
                    continue
                m = self.meta[idx]
                hits.append({
                    "doc_id": m.get("doc_id"),
                    "score": float(score),
                    "snippet": m.get("snippet", ""),
                    "source": m.get("source")
                })
            return hits

        # Fallback: TF-IDF if embedding model fails to load
        self._ensure_tfidf()
        return self._search_tfidf(query, top_k=top_k)
