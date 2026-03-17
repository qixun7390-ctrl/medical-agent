import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data" / "eval" / "generation_preds.jsonl"

if not PRED.exists():
    print("Missing preds")
    raise SystemExit(1)

rows = [json.loads(x) for x in PRED.read_text(encoding="utf-8").splitlines() if x.strip()]

# Simple character-level F1 for Chinese text

def char_f1(pred: str, gold: str):
    pred = re.sub(r"\s+", "", pred)
    gold = re.sub(r"\s+", "", gold)
    if not pred or not gold:
        return 0.0
    p = list(pred)
    g = list(gold)
    inter = len(set(p) & set(g))
    precision = inter / max(len(set(p)), 1)
    recall = inter / max(len(set(g)), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# Citation coverage: percentage of answers with at least one [doc_id]
# Faithfulness proxy: overlap between answer and evidence snippets

citation_hits = 0
faith_scores = []
f1_scores = []
latencies = []
errors = 0

for r in rows:
    if "error" in r:
        errors += 1
        continue
    ans = r.get("answer", "")
    gold = r.get("gold", "")
    evidences = r.get("evidences", [])
    lat = r.get("latency_ms", 0)
    latencies.append(lat)

    # citations
    if re.search(r"\[[^\]]+\]", ans):
        citation_hits += 1

    # faithfulness proxy
    overlap = 0.0
    for ev in evidences:
        snippet = ev.get("snippet", "")
        if snippet:
            inter = len(set(ans) & set(snippet))
            overlap = max(overlap, inter / max(len(set(snippet)), 1))
    faith_scores.append(overlap)

    # QA f1
    if gold:
        f1_scores.append(char_f1(ans, gold))

n = len(rows) - errors

print(json.dumps({
    "count": n,
    "errors": errors,
    "citation_coverage": citation_hits / n if n else 0,
    "faithfulness_proxy": sum(faith_scores) / n if n else 0,
    "qa_char_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
    "latency_p50_ms": sorted(latencies)[int(0.5*len(latencies))] if latencies else 0,
    "latency_p95_ms": sorted(latencies)[int(0.95*len(latencies))-1] if latencies else 0
}, ensure_ascii=False, indent=2))

out = ROOT / "data" / "eval" / "generation_metrics.json"
out.write_text(json.dumps({
    "count": n,
    "errors": errors,
    "citation_coverage": citation_hits / n if n else 0,
    "faithfulness_proxy": sum(faith_scores) / n if n else 0,
    "qa_char_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
    "latency_p50_ms": sorted(latencies)[int(0.5*len(latencies))] if latencies else 0,
    "latency_p95_ms": sorted(latencies)[int(0.95*len(latencies))-1] if latencies else 0
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {out}")
