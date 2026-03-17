import json
from pathlib import Path
from collections import Counter
import networkx as nx
from app.utils.text import extract_terms

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "data" / "processed" / "qa_corpus.jsonl"
OUT = ROOT / "data" / "processed" / "kg.graph.json"

G = nx.Graph()
term_freq = Counter()

if QA.exists():
    for line in QA.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        text = obj.get("text", "")
        # Expect format: 问：...\n答：...
        parts = text.split("\n")
        q = parts[0] if parts else ""
        a = parts[1] if len(parts) > 1 else ""
        q_terms = extract_terms(q)
        a_terms = extract_terms(a)
        for t in q_terms + a_terms:
            term_freq[t] += 1
        # connect question terms to answer terms
        for qt in q_terms:
            for at in a_terms:
                if qt != at:
                    if G.has_edge(qt, at):
                        G[qt][at]["weight"] += 1
                    else:
                        G.add_edge(qt, at, weight=1, rel="related")

# add nodes with frequency as weight
for t, freq in term_freq.items():
    if not G.has_node(t):
        G.add_node(t)
    G.nodes[t]["label"] = t
    G.nodes[t]["freq"] = freq

nodes = [{"id": n, "attrs": dict(G.nodes[n])} for n in G.nodes]
edges = [{"source": u, "target": v, "attrs": dict(G[u][v])} for u, v in G.edges]

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False), encoding="utf-8")

print(f"KG built: {len(nodes)} nodes, {len(edges)} edges -> {OUT}")
