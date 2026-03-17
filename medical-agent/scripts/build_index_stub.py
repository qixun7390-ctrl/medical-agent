import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "processed" / "corpus.jsonl"
META = ROOT / "data" / "processed" / "meta.jsonl"

META.parent.mkdir(parents=True, exist_ok=True)

with CORPUS.open("r", encoding="utf-8") as f_in, META.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        obj = json.loads(line)
        snippet = obj["text"][:200].replace("\n", " ")
        f_out.write(json.dumps({"doc_id": obj["doc_id"], "snippet": snippet}) + "\n")

print(f"Wrote metadata to {META}")
