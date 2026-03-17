import os
import json
from typing import Dict, Any, List, Tuple
from pathlib import Path


_CACHE = {
    "mtime": None,
    "numeric_by_slot": {},
    "keyword_rules": [],
    "keyword_to_rules": {},
    "aho": None,
}


def _read_rules(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {"rules": []}
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    try:
        return json.loads(text) if text.strip() else {"rules": []}
    except Exception:
        return {"rules": []}


def _match_numeric(value: float, op: str, threshold: float) -> bool:
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    if op == ">":
        return value > threshold
    if op == "<":
        return value < threshold
    if op == "==":
        return value == threshold
    return False


def _build_cache(rules_path: str) -> None:
    mtime = os.path.getmtime(rules_path) if os.path.exists(rules_path) else None
    if _CACHE["mtime"] == mtime:
        return

    data = _read_rules(rules_path)
    rules = data.get("rules", []) if isinstance(data, dict) else []

    numeric_by_slot: Dict[str, List[Dict[str, Any]]] = {}
    keyword_rules: List[Dict[str, Any]] = []
    keyword_to_rules: Dict[str, List[Dict[str, Any]]] = {}

    for r in rules:
        rtype = r.get("type")
        if rtype == "numeric":
            slot = r.get("slot")
            if slot:
                numeric_by_slot.setdefault(slot, []).append(r)
        elif rtype == "keyword":
            keywords = r.get("keywords", [])
            if keywords:
                keyword_rules.append(r)
                for k in keywords:
                    keyword_to_rules.setdefault(k, []).append(r)

    aho = None
    try:
        import ahocorasick  # optional
        aho = ahocorasick.Automaton()
        for kw in keyword_to_rules.keys():
            aho.add_word(kw, kw)
        aho.make_automaton()
    except Exception:
        aho = None

    _CACHE["mtime"] = mtime
    _CACHE["numeric_by_slot"] = numeric_by_slot
    _CACHE["keyword_rules"] = keyword_rules
    _CACHE["keyword_to_rules"] = keyword_to_rules
    _CACHE["aho"] = aho


def _match_keyword_rules(text: str) -> List[Dict[str, Any]]:
    matched = []
    seen = set()

    if _CACHE["aho"] is not None:
        for _, kw in _CACHE["aho"].iter(text):
            for r in _CACHE["keyword_to_rules"].get(kw, []):
                rid = id(r)
                if rid in seen:
                    continue
                seen.add(rid)
                matched.append(r)
        return matched

    # Fallback: naive contains, but only scan keyword rules
    for r in _CACHE["keyword_rules"]:
        keywords = r.get("keywords", [])
        if any(k in text for k in keywords):
            matched.append(r)
    return matched


def apply_rules(text: str, slots: Dict[str, Any], rules_path: str) -> Dict[str, Any]:
    _build_cache(rules_path)

    add_queries: List[str] = []
    guidance: List[str] = []
    hits: List[str] = []
    evidence: List[Dict[str, Any]] = []

    # Numeric rules: only check slots that exist
    for slot, rules in _CACHE["numeric_by_slot"].items():
        if slot not in slots:
            continue
        try:
            val = float(slots[slot])
        except Exception:
            continue
        for r in rules:
            try:
                if _match_numeric(val, r.get("op"), float(r.get("value"))):
                    add_queries.extend(r.get("add_queries", []))
                    if r.get("guidance"):
                        guidance.append(r.get("guidance"))
                    if r.get("name"):
                        hits.append(r.get("name"))
                        if r.get("evidence"):
                            evidence.append({
                                "doc_id": f"rule:{r.get('name')}",
                                "score": 0.7,
                                "snippet": r.get("evidence"),
                                "source": "; ".join(r.get("sources", [])) if r.get("sources") else "rules"
                            })
            except Exception:
                continue

    # Keyword rules: fast match
    for r in _match_keyword_rules(text):
        add_queries.extend(r.get("add_queries", []))
        if r.get("guidance"):
            guidance.append(r.get("guidance"))
        if r.get("name"):
            hits.append(r.get("name"))
            if r.get("evidence"):
                evidence.append({
                    "doc_id": f"rule:{r.get('name')}",
                    "score": 0.7,
                    "snippet": r.get("evidence"),
                    "source": "; ".join(r.get("sources", [])) if r.get("sources") else "rules"
                })

    return {"add_queries": add_queries, "guidance": guidance, "hits": hits, "evidence": evidence}
