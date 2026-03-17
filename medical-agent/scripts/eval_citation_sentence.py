import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data" / "eval" / "generation_preds.jsonl"

if not PRED.exists():
    print("Missing preds")
    raise SystemExit(1)

rows = [json.loads(x) for x in PRED.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]

# Sentence-level citation coverage

def sentence_citation_coverage(text: str):
    # split by Chinese/English sentence enders
    sents = re.split(r"(?<=[。！？!?])", text)
    sents = [s.strip() for s in sents if s.strip()]
    if not sents:
        return 0.0
    ok = 0
    for s in sents:
        if re.search(r"\[[^\]]+\]", s):
            ok += 1
    return ok / len(sents)

citation_rates = []

for r in rows:
    if "error" in r:
        continue
    ans = r.get("answer", "")
    citation_rates.append(sentence_citation_coverage(ans))

if not citation_rates:
    print("No valid answers")
    raise SystemExit(1)

avg_rate = sum(citation_rates) / len(citation_rates)

out = ROOT / "data" / "eval" / "citation_sentence_metrics.json"
out.write_text(json.dumps({"sentence_citation_coverage": avg_rate}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"sentence_citation_coverage": avg_rate}, ensure_ascii=False, indent=2))
