from app.storage.vector_store import VectorStore
from app.kg.graph_store import GraphStore
from app.core.config import settings
from app.storage.cache_store import CacheStore
import os
import json
from pathlib import Path

class Retriever:
    def __init__(self):
        # Vector/KG backends
        self.vs = VectorStore()
        self.kg = GraphStore(settings.kg_path)

        # Optional precomputed retrieval (used during eval to avoid heavy deps)
        self.use_precomputed = os.environ.get("USE_PRECOMPUTED_RETRIEVAL", "0") == "1"
        self.precomputed = {}
        self.meta = {}
        if self.use_precomputed:
            path = os.environ.get("PRECOMPUTED_RETRIEVAL_PATH", "")
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        self.precomputed[obj["question"]] = obj.get("pred", [])
            if os.path.exists(settings.vector_meta_path):
                with open(settings.vector_meta_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        m = json.loads(line)
                        self.meta[m["doc_id"]] = m

        # Hot query cache (Redis or in-memory fallback)
        self.cache = CacheStore(settings.redis_url, disabled=not settings.cache_enabled, name="retrieval")

    async def retrieve(self, query: str):
        # 1) Cache hit
        cache_key = f"retrieval:{query}"
        cached = await self.cache.get(cache_key)
        if cached:
            try:
                ids = json.loads(cached)
                hits = []
                for doc_id in ids:
                    m = self.meta.get(doc_id, {})
                src = m.get("source")
                if src:
                    src = Path(str(src)).name
                hits.append({"doc_id": doc_id, "score": 1.0, "snippet": m.get("snippet", ""), "source": src})
                return hits
            except Exception:
                pass

        # 2) Precomputed retrieval (eval mode)
        if self.use_precomputed:
            ids = self.precomputed.get(query, [])[:20]
            if ids:
                await self.cache.set(cache_key, json.dumps(ids), ttl=settings.cache_ttl_seconds)
                hits = []
                for doc_id in ids:
                    m = self.meta.get(doc_id, {})
                src = m.get("source")
                if src:
                    src = Path(str(src)).name
                hits.append({"doc_id": doc_id, "score": 1.0, "snippet": m.get("snippet", ""), "source": src})
                return hits
            # If no precomputed hit, fall back to live retrieval

        # 3) Live retrieval (vector + KG)
        vec_hits = self.vs.search(query, top_k=20)
        for h in vec_hits:
            src = h.get("source")
            if src:
                h["source"] = Path(str(src)).name
        kg_hits = self.kg.search(query) if settings.retrieval_mode != "qa_question" else []
        hits = vec_hits + kg_hits

        # Cache top doc_ids for hot queries
        await self.cache.set(cache_key, json.dumps([h.get("doc_id") for h in hits[:10]]), ttl=settings.cache_ttl_seconds)
        return hits
