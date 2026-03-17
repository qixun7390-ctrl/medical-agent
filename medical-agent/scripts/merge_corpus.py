import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "processed" / "raw_corpus.jsonl"
QA = ROOT / "data" / "processed" / "qa_corpus.jsonl"
OUT = ROOT / "data" / "processed" / "corpus.jsonl"

OUT.parent.mkdir(parents=True, exist_ok=True)

count = 0
with OUT.open("w", encoding="utf-8") as f_out:
    for p in [RAW, QA]:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            f_out.write(line + "\n")
            count += 1

print(f"Merged corpus size: {count} -> {OUT}")
