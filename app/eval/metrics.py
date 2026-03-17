import json
from pathlib import Path


def eval_recall(gold_path: str, pred_path: str):
    gold = [json.loads(x) for x in Path(gold_path).read_text(encoding="utf-8").splitlines()]
    pred = [json.loads(x) for x in Path(pred_path).read_text(encoding="utf-8").splitlines()]
    # Placeholder metrics
    return {"recall@5": 0.0, "ndcg@5": 0.0}
