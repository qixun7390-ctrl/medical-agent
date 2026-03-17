import json
import os
from pathlib import Path
import re
import httpx

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data" / "eval" / "generation_preds.jsonl"

LLM_ENDPOINT = os.environ.get("EVAL_LLM_ENDPOINT", "http://localhost:8001/v1")
LLM_MODEL = os.environ.get("EVAL_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
LLM_TIMEOUT = float(os.environ.get("EVAL_LLM_TIMEOUT", "20"))

PROMPT_TEMPLATE = """
You are a strict medical QA evaluator.
Given a user question, an answer, and evidence snippets, judge:
1) Faithfulness: Is every key claim in the answer supported by the evidence?
2) Citation coverage: Does the answer cite evidence for each key claim?

Return JSON only in this format:
{{
  "faithful": true/false,
  "citation_ok": true/false,
  "issues": ["..."]
}}

Question: {question}
Answer: {answer}
Evidence:
{evidence}
"""

async def llm_judge(question, answer, evidence):
    prompt = PROMPT_TEMPLATE.format(question=question, answer=answer, evidence=evidence)
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        r = await client.post(f"{LLM_ENDPOINT}/chat/completions", json=payload)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        # try to extract json
        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            return {"faithful": False, "citation_ok": False, "issues": ["invalid_json"]}
        return json.loads(match.group(0))

import asyncio

async def main():
    if not PRED.exists():
        print("Missing preds")
        raise SystemExit(1)

    rows = [json.loads(x) for x in PRED.read_text(encoding="utf-8").splitlines() if x.strip()]
    faithful = 0
    citation_ok = 0
    total = 0

    for r in rows:
        if "error" in r:
            continue
        question = r.get("question", "")
        answer = r.get("answer", "")
        evidences = r.get("evidences", [])
        evidence_text = "\n".join([f"[{e.get('doc_id')}] {e.get('snippet','')}" for e in evidences])
        judge = await llm_judge(question, answer, evidence_text)
        if judge.get("faithful"):
            faithful += 1
        if judge.get("citation_ok"):
            citation_ok += 1
        total += 1

    print(json.dumps({
        "count": total,
        "faithfulness_rate": faithful / total if total else 0,
        "citation_coverage_rate": citation_ok / total if total else 0
    }, ensure_ascii=False, indent=2))

    out = ROOT / "data" / "eval" / "faithfulness_metrics.json"
    out.write_text(json.dumps({
        "count": total,
        "faithfulness_rate": faithful / total if total else 0,
        "citation_coverage_rate": citation_ok / total if total else 0
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")

if __name__ == "__main__":
    asyncio.run(main())
