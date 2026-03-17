import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eval" / "retrieval_metrics.json"

# placeholder for later: parse eval_retrieval.py output into file
OUT.write_text(json.dumps({"recall@5": 0, "recall@10": 0, "ndcg@10": 0}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {OUT}")
