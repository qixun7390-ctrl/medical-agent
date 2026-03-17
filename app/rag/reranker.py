import os
from app.core.config import settings


class Reranker:
    def __init__(self):
        self.model = None
        self._loaded = False

    def _ensure_model(self):
        if self._loaded:
            return
        from sentence_transformers import CrossEncoder
        device = os.environ.get("RERANKER_DEVICE", "auto")
        kwargs = {}
        if device in {"cpu", "cuda"}:
            kwargs["device"] = device
        self.model = CrossEncoder(settings.reranker_model, **kwargs)
        self._loaded = True

    async def rerank(self, query: str, candidates: list):
        if not candidates:
            return []
        self._ensure_model()
        pairs = [(query, c.get("snippet", "")) for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["score"] = float(s)
        return sorted(candidates, key=lambda x: x["score"], reverse=True)[:8]
