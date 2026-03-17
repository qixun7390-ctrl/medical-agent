import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eval" / "faithfulness_metrics.json"

# placeholder for later: parse eval_faithfulness_llm.py output into file
OUT.write_text(json.dumps({"faithfulness_rate": 0, "citation_coverage_rate": 0}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {OUT}")
