import asyncio
import json
import os
from pathlib import Path
from app.agent.coordinator import AgentCoordinator
from app.models.schemas import ChatRequest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval" / "qa_eval.jsonl"
OUT = ROOT / "data" / "eval" / "generation_preds.jsonl"
SAMPLE = int(os.environ.get("GEN_EVAL_SAMPLE", "300"))

async def main():
    if not EVAL.exists():
        print("Missing eval set")
        return

    rows = [json.loads(x) for x in EVAL.read_text(encoding="utf-8").splitlines() if x.strip()]
    if SAMPLE < len(rows):
        rows = rows[:SAMPLE]

    agent = AgentCoordinator()
    with OUT.open("w", encoding="utf-8") as f_out:
        for obj in rows:
            q = obj["question"]
            gold = obj.get("answer", "")
            req = ChatRequest(user_id="eval", session_id="eval", query=q, history=[])
            try:
                res = await agent.run(req)
                f_out.write(json.dumps({
                    "question": q,
                    "gold": gold,
                    "answer": res.answer,
                    "evidences": [e.model_dump() for e in res.evidences],
                    "hypotheses": [h.model_dump() for h in res.hypotheses],
                    "guidance": res.guidance,
                    "latency_ms": res.latency_ms,
                }, ensure_ascii=False) + "\n")
            except Exception as e:
                f_out.write(json.dumps({
                    "question": q,
                    "gold": gold,
                    "error": str(e)
                }, ensure_ascii=False) + "\n")

    print(f"Wrote preds to {OUT}")

if __name__ == "__main__":
    asyncio.run(main())
