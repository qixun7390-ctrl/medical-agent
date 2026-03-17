import json
from pathlib import Path
from app.rag.retriever import Retriever
import asyncio

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval" / "qa_eval.jsonl"
PRED = ROOT / "data" / "eval" / "retrieval_preds.jsonl"

async def main():
    ret = Retriever()
    if not EVAL.exists():
        print("Missing eval set")
        return
    with EVAL.open("r", encoding="utf-8") as f_in, PRED.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            obj = json.loads(line)
            q = obj["question"]
            doc_id = obj["doc_id"]
            hits = await ret.retrieve(q)
            pred_ids = [h["doc_id"] for h in hits[:10]]
            f_out.write(json.dumps({"question": q, "gold": doc_id, "pred": pred_ids}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    asyncio.run(main())
