import json
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data" / "eval" / "retrieval_preds.jsonl"

if not PRED.exists():
    print("Missing preds")
    raise SystemExit(1)

rows = [json.loads(x) for x in PRED.read_text(encoding="utf-8").splitlines() if x.strip()]

recall_at_5 = 0
recall_at_10 = 0
ndcg_at_10 = 0

for r in rows:
    gold = r["gold"]
    pred = r["pred"]
    if gold in pred[:5]:
        recall_at_5 += 1
    if gold in pred[:10]:
        recall_at_10 += 1
    if gold in pred:
        rank = pred.index(gold) + 1
        ndcg_at_10 += 1 / math.log2(rank + 1)

n = len(rows)
print(json.dumps({
    "recall@5": recall_at_5 / n if n else 0,
    "recall@10": recall_at_10 / n if n else 0,
    "ndcg@10": ndcg_at_10 / n if n else 0
}, ensure_ascii=False, indent=2))

out = ROOT / "data" / "eval" / "retrieval_metrics.json"
out.write_text(json.dumps({
    "recall@5": recall_at_5 / n if n else 0,
    "recall@10": recall_at_10 / n if n else 0,
    "ndcg@10": ndcg_at_10 / n if n else 0
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {out}")
