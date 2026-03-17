import json
from pathlib import Path
import networkx as nx

class GraphStore:
    def __init__(self, path: str = "data/processed/kg.graph.json"):
        self.path = Path(path)
        self.graph = nx.Graph()
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        for n in data.get("nodes", []):
            self.graph.add_node(n["id"], **n.get("attrs", {}))
        for e in data.get("edges", []):
            self.graph.add_edge(e["source"], e["target"], **e.get("attrs", {}))

    def search(self, query: str):
        # naive keyword match against node labels
        hits = []
        for n, attrs in self.graph.nodes(data=True):
            label = attrs.get("label", "")
            if query in label:
                hits.append({
                    "doc_id": f"kg:{n}",
                    "score": 0.8,
                    "snippet": f"KG Node: {label}",
                    "source": "kg"
                })
        return hits[:5]
