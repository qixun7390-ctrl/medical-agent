import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "data" / "processed" / "qa_corpus.jsonl"
KG = ROOT / "data" / "processed" / "kg.graph.json"

nodes = []
edges = []

if QA.exists():
    for line in QA.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        text = obj["text"]
        nodes.append({"id": obj["doc_id"], "attrs": {"label": text[:80]}})

# naive random edges
random.seed(42)
for i in range(min(len(nodes), 200)):
    if i + 1 < len(nodes):
        edges.append({"source": nodes[i]["id"], "target": nodes[i+1]["id"], "attrs": {"rel": "related"}})

KG.parent.mkdir(parents=True, exist_ok=True)
KG.write_text(json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False), encoding="utf-8")

print(f"KG built with {len(nodes)} nodes and {len(edges)} edges")
