import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eval" / "generation_metrics.json"

# placeholder for later: parse eval_generation.py output into file
OUT.write_text(json.dumps({"citation_coverage": 0, "faithfulness_proxy": 0, "qa_char_f1": 0}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {OUT}")
