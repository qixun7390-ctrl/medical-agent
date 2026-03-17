import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "processed" / "corpus.jsonl"
INDEX = ROOT / "data" / "processed" / "faiss.index"
META = ROOT / "data" / "processed" / "meta.jsonl"

model = SentenceTransformer("BAAI/bge-m3")

texts = []
meta = []
with CORPUS.open("r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        text = obj["text"]
        texts.append(text)
        meta.append({
            "doc_id": obj["doc_id"],
            "snippet": text[:200].replace("\n", " "),
            "source": obj.get("source")
        })

embs = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
embs = np.array(embs, dtype="float32")

index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)
faiss.write_index(index, str(INDEX))

with META.open("w", encoding="utf-8") as f:
    for m in meta:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")

print(f"Indexed {len(meta)} documents -> {INDEX}")
