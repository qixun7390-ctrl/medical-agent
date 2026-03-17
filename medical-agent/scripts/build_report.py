import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data" / "eval" / "report.json"

# Merge retrieval + generation metrics if present
metrics = {}
for p in [ROOT / "data" / "eval" / "retrieval_metrics.json", ROOT / "data" / "eval" / "generation_metrics.json", ROOT / "data" / "eval" / "faithfulness_metrics.json"]:
    if p.exists():
        metrics.update(json.loads(p.read_text(encoding="utf-8")))

REPORT.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote report -> {REPORT}")
