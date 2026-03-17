import json
import os
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "processed" / "qa_corpus.jsonl"
INDEX = ROOT / "data" / "processed" / "faiss_qa.index"
META = ROOT / "data" / "processed" / "meta_qa.jsonl"

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
BATCH_SIZE = int(os.environ.get("EMBED_BATCH", "64"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "5000"))

model = SentenceTransformer(MODEL_NAME)

index = None
count = 0
buffer_texts = []
buffer_meta = []


def _flush_batch():
    global index, count, buffer_texts, buffer_meta
    if not buffer_texts:
        return
    embs = model.encode(buffer_texts, normalize_embeddings=True, batch_size=BATCH_SIZE, show_progress_bar=False)
    embs = np.array(embs, dtype="float32")
    if index is None:
        index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    with META.open("a", encoding="utf-8") as f_meta:
        for m in buffer_meta:
            f_meta.write(json.dumps(m, ensure_ascii=False) + "\n")
    count += len(buffer_texts)
    buffer_texts = []
    buffer_meta = []
    if count % LOG_EVERY == 0:
        print(f"Indexed {count} docs...")


META.write_text("", encoding="utf-8")

with CORPUS.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        text = obj["text"]
        buffer_texts.append(text)
        buffer_meta.append({
            "doc_id": obj["doc_id"],
            "snippet": text[:200].replace("\n", " "),
            "source": obj.get("source")
        })
        if len(buffer_texts) >= BATCH_SIZE:
            _flush_batch()

_flush_batch()

if index is None:
    raise RuntimeError("No documents found to index.")

faiss.write_index(index, str(INDEX))
print(f"Indexed {count} QA docs -> {INDEX}")
