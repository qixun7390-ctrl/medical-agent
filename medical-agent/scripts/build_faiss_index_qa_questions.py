import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "processed" / "qa_corpus.jsonl"
INDEX = ROOT / "data" / "processed" / "faiss_qa_q.index"
META = ROOT / "data" / "processed" / "meta_qa_q.jsonl"

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

texts = []
meta = []
with CORPUS.open("r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        text = obj["text"]
        # extract question part
        q = text.split("\n")[0].replace("问：", "").strip()
        if not q:
            continue
        texts.append(q)
        meta.append({
            "doc_id": obj["doc_id"],
            "snippet": q[:200],
            "source": obj.get("source")
        })

embs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
embs = np.array(embs, dtype="float32")

index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)
faiss.write_index(index, str(INDEX))

with META.open("w", encoding="utf-8") as f:
    for m in meta:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")

print(f"Indexed {len(meta)} QA questions -> {INDEX}")
