import json
import os
from pathlib import Path
import random
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "processed" / "corpus.jsonl"
INDEX = Path(os.environ.get("VECTOR_INDEX_PATH", str(ROOT / "data" / "processed" / "faiss_fast.index")))
META = Path(os.environ.get("VECTOR_META_PATH", str(ROOT / "data" / "processed" / "meta_fast.jsonl")))
SAMPLE = int(os.environ.get("INDEX_SAMPLE", "20000"))
MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

model = SentenceTransformer(MODEL)

rows = [json.loads(x) for x in CORPUS.read_text(encoding="utf-8").splitlines() if x.strip()]
if SAMPLE < len(rows):
    random.seed(42)
    rows = random.sample(rows, SAMPLE)

texts = [r["text"] for r in rows]
meta = [{"doc_id": r["doc_id"], "snippet": r["text"][:200].replace("\n", " "), "source": r.get("source")} for r in rows]

embs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
embs = np.array(embs, dtype="float32")

index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)
faiss.write_index(index, str(INDEX))

META.parent.mkdir(parents=True, exist_ok=True)
with META.open("w", encoding="utf-8") as f:
    for m in meta:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")

print(f"Indexed {len(meta)} docs with {MODEL} -> {INDEX}")
