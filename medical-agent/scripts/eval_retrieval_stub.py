import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval" / "qa_eval.jsonl"
PRED = ROOT / "data" / "eval" / "retrieval_preds.jsonl"

# Placeholder evaluator

def load(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

if not EVAL.exists() or not PRED.exists():
    print("Missing eval or pred files")
    raise SystemExit(1)

# placeholder metrics
print(json.dumps({"recall@5": 0.0, "ndcg@5": 0.0}, ensure_ascii=False))
