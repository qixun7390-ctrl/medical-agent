from collections import defaultdict

_stats = defaultdict(lambda: {"hit": 0, "miss": 0})

def mark_hit(name: str):
    _stats[name]["hit"] += 1

def mark_miss(name: str):
    _stats[name]["miss"] += 1

def get_stats():
    out = {k: dict(v) for k, v in _stats.items()}
    total_hit = sum(v["hit"] for v in out.values())
    total_miss = sum(v["miss"] for v in out.values())
    out["total"] = {"hit": total_hit, "miss": total_miss}
    return out
