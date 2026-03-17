import os
import json
from collections import Counter, defaultdict
import asyncio
import sys

# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.storage.cache_store import CacheStore
from app.core.config import settings


def extract_question(text: str) -> str:
    # Expect "问：xxx\n答：yyy" format; fallback to full line
    if "问：" in text:
        line = text.split("\n")[0]
        return line.replace("问：", "").strip()
    return text.strip()


async def main():
    corpus_path = os.environ.get(
        "PREWARM_CORPUS",
        os.path.join(ROOT, "medical-agent", "data", "processed", "qa_corpus.jsonl"),
    )
    top_n = int(os.environ.get("PREWARM_TOPN", "200"))
    top_k_docs = int(os.environ.get("PREWARM_TOPK_DOCS", "10"))

    if not os.path.exists(corpus_path):
        print(f"Corpus not found: {corpus_path}")
        return

    # Count question frequency + collect doc_ids per question
    counter = Counter()
    doc_map = defaultdict(list)
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            q = extract_question(obj.get("text", ""))
            if not q:
                continue
            doc_id = obj.get("doc_id")
            if doc_id:
                doc_map[q].append(doc_id)
            counter[q] += 1

    top_questions = [q for q, _ in counter.most_common(top_n)]
    print(f"Prewarming {len(top_questions)} questions (retrieval-only, no embeddings)")

    cache = CacheStore(settings.redis_url, disabled=not settings.cache_enabled, name="retrieval")
    for q in top_questions:
        ids = doc_map.get(q, [])[:top_k_docs]
        if not ids:
            continue
        await cache.set(f"retrieval:{q}", json.dumps(ids), ttl=settings.cache_ttl_seconds)

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
