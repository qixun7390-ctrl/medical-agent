import json
import os
import random
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval" / "qa_eval.jsonl"
PRED = ROOT / "data" / "eval" / "retrieval_preds.jsonl"

INDEX = Path(os.environ.get("VECTOR_INDEX_PATH", str(ROOT / "data" / "processed" / "faiss_fast.index")))
META = Path(os.environ.get("VECTOR_META_PATH", str(ROOT / "data" / "processed" / "meta_fast.jsonl")))
MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
SAMPLE = int(os.environ.get("EVAL_SAMPLE", "300"))
TOPK = int(os.environ.get("EVAL_TOPK", "10"))

if not EVAL.exists():
    raise SystemExit("Missing eval set")

rows = [json.loads(x) for x in EVAL.read_text(encoding="utf-8").splitlines() if x.strip()]
if SAMPLE < len(rows):
    random.seed(42)
    rows = random.sample(rows, SAMPLE)

questions = [r["question"] for r in rows]

gold_ids = [r["doc_id"] for r in rows]

index = faiss.read_index(str(INDEX))
meta = [json.loads(l) for l in META.read_text(encoding="utf-8").splitlines() if l.strip()]

model = SentenceTransformer(MODEL)
q_emb = model.encode(questions, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
q_emb = np.array(q_emb, dtype="float32")

scores, idxs = index.search(q_emb, TOPK)

with PRED.open("w", encoding="utf-8") as f_out:
    for q, gold, idx_list in zip(questions, gold_ids, idxs):
        pred_ids = []
        for idx in idx_list:
            if idx < 0 or idx >= len(meta):
                continue
            pred_ids.append(meta[idx]["doc_id"])
        f_out.write(json.dumps({"question": q, "gold": gold, "pred": pred_ids}, ensure_ascii=False) + "\n")

print(f"Wrote preds for {len(rows)} queries -> {PRED}")
