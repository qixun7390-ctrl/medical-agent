import os
import json
import sys
from pathlib import Path
import httpx
import gradio as gr

# Ensure project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.kg.graph_store import GraphStore
from app.core.config import settings

# Endpoints
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "http://127.0.0.1:8001/v1")
MODEL_ID = os.environ.get("LLM_MODEL", "/models/Qwen2.5-1.5B-Instruct")
AGENT_ENDPOINT = os.environ.get("AGENT_ENDPOINT", "http://127.0.0.1:8000/api/chat")
AGENT_STREAM_ENDPOINT = os.environ.get("AGENT_STREAM_ENDPOINT", "http://127.0.0.1:8000/api/chat_stream")
CACHE_ENDPOINT = os.environ.get("CACHE_ENDPOINT", "http://127.0.0.1:8000/api/cache_stats")
USE_STREAM = os.environ.get("USE_STREAM", "1") == "1"

KG = GraphStore(settings.kg_path)

# Optional precomputed retrieval (eval mode)
USE_PRECOMPUTED = os.environ.get("USE_PRECOMPUTED_RETRIEVAL", "0") == "1"
PRECOMPUTED_PATH = os.environ.get("PRECOMPUTED_RETRIEVAL_PATH", "")
PRECOMPUTED = {}
if USE_PRECOMPUTED and PRECOMPUTED_PATH and os.path.exists(PRECOMPUTED_PATH):
    with open(PRECOMPUTED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            PRECOMPUTED[obj["question"]] = obj.get("pred", [])

async def call_agent_api(message):
    payload = {"user_id": "ui", "session_id": "ui", "query": message, "history": []}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(AGENT_ENDPOINT, json=payload)
        r.raise_for_status()
        return r.json()

async def stream_agent_api(message):
    payload = {"user_id": "ui", "session_id": "ui", "query": message, "history": []}
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", AGENT_STREAM_ENDPOINT, json=payload) as r:
            r.raise_for_status()
            current_event = None
            data_lines = []
            async for line in r.aiter_lines():
                if line.startswith("event:"):
                    current_event = line[len("event:"):].strip()
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[len("data:"):].strip())
                    continue
                if line == "":
                    if current_event and data_lines:
                        data = "\n".join(data_lines)
                        yield (current_event, data)
                    current_event = None
                    data_lines = []
            # flush tail if any
            if current_event and data_lines:
                data = "\n".join(data_lines)
                yield (current_event, data)

async def call_vllm(message):
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": "You are a medical assistant. Use evidence when possible and advise seeking care if needed."},
            {"role": "user", "content": message},
        ],
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{LLM_ENDPOINT}/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def fetch_cache_stats():
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(CACHE_ENDPOINT)
        r.raise_for_status()
        return json.dumps(r.json(), ensure_ascii=False)

# Main UI handler
ALLOW_RAW_FALLBACK = os.environ.get("ALLOW_RAW_LLM_FALLBACK", "0") == "1"

async def chat_fn(message, history):
    answer = ""
    evidences = []
    hypotheses = []
    guidance = ""

    # Prefer agent API; optional fallback to raw vLLM
    try:
        res = await call_agent_api(message)
        answer = res.get("answer", "")
        evidences = res.get("evidences", [])
        hypotheses = res.get("hypotheses", [])
        guidance = res.get("guidance", "")
    except Exception as e:
        if ALLOW_RAW_FALLBACK:
            answer = await call_vllm(message)
        else:
            answer = "后端服务不可用，未能生成基于证据的回答。请确认 8000 端口的 Agent 服务已启动。"
            evidences = [{"doc_id": "error", "score": 0.0, "snippet": f"Agent API error: {e}", "source": "client"}]

    kg_hits = KG.search(message)

    if USE_PRECOMPUTED:
        pred_ids = PRECOMPUTED.get(message, [])
        evidences = evidences or [{"doc_id": x, "score": 1.0, "snippet": "(precomputed)", "source": "precomputed"} for x in pred_ids[:5]]

    evidence_text = "\n".join([f"[{e.get('doc_id')}] {e.get('snippet','')}" for e in evidences])
    hypothesis_text = "\n".join([f"- {h.get('title')} (conf: {h.get('confidence')})" for h in hypotheses])
    kg_text = "\n".join([f"- {h.get('snippet','')}" for h in kg_hits])
    rule_hits = res.get("rule_hits", []) if isinstance(res, dict) else []
    rule_text = "\n".join([f"- {r}" for r in rule_hits])

    cache_stats = "(cache stats unavailable)"
    try:
        cache_stats = await fetch_cache_stats()
    except Exception:
        pass

    return (
        answer,
        evidence_text or "(no evidence)",
        hypothesis_text or "(no hypotheses)",
        guidance or "(none)",
        kg_text or "(no KG hit)",
        rule_text or "(no rule hit)",
        cache_stats,
    )

async def chat_fn_stream(message, history):
    # Streaming handler (SSE): evidence first, then answer chunks
    answer = ""
    evidences = []
    hypotheses = []
    guidance = ""
    rule_text = ""
    evidence_text = "(no evidence)"
    kg_text = "\n".join([f"- {h.get('snippet','')}" for h in KG.search(message)])
    cache_stats = "(cache stats unavailable)"

    try:
        async for ev, data in stream_agent_api(message):
            if data is None:
                continue
            try:
                payload = json.loads(data)
            except Exception:
                payload = data

            if ev == "evidence":
                evidences = payload
                evidence_text = "\n".join([f"[{e.get('doc_id')}] {e.get('snippet','')}" for e in evidences])
            elif ev == "rule_hits":
                rule_text = "\n".join([f"- {r}" for r in payload])
            elif ev == "answer_chunk":
                answer += payload
            elif ev == "error":
                answer += f"\n[Error] {payload}"

            yield (
                answer,
                evidence_text or "(no evidence)",
                "(streaming)" if not hypotheses else "\n".join([f"- {h.get('title')} (conf: {h.get('confidence')})" for h in hypotheses]),
                guidance or "(none)",
                kg_text or "(no KG hit)",
                rule_text or "(no rule hit)",
                cache_stats,
            )

        # final cache stats
        try:
            cache_stats = await fetch_cache_stats()
        except Exception:
            pass
        yield (
            answer,
            evidence_text or "(no evidence)",
            "(streaming)",
            guidance or "(none)",
            kg_text or "(no KG hit)",
            rule_text or "(no rule hit)",
            cache_stats,
        )
    except Exception as e:
        # Fallback to non-streaming call if stream fails
        try:
            res = await call_agent_api(message)
            answer = res.get("answer", "")
            evidences = res.get("evidences", [])
            guidance = res.get("guidance", "")
            evidence_text = "\n".join([f"[{e.get('doc_id')}] {e.get('snippet','')}" for e in evidences])
            yield (
                answer,
                evidence_text or "(no evidence)",
                "(no hypotheses)",
                guidance or "(none)",
                kg_text or "(no KG hit)",
                rule_text or "(no rule hit)",
                cache_stats,
            )
        except Exception:
            yield (
                "后端服务不可用，未能生成基于证据的回答。请确认 8000 端口的 Agent 服务已启动。",
                f"[error] {e}",
                "(no hypotheses)",
                "(none)",
                kg_text or "(no KG hit)",
                rule_text or "(no rule hit)",
                cache_stats,
            )

with gr.Blocks(title="Medical Agent (vLLM + KG)") as demo:
    gr.Markdown("# Medical Agent Demo\n- vLLM + KG\n- Evidence & Hypotheses panel + Cache Stats")

    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="User Question")
            btn = gr.Button("Ask")
        with gr.Column():
            out = gr.Textbox(label="Answer")

    with gr.Row():
        evidence_box = gr.Textbox(label="Evidence", lines=8)
        hypothesis_box = gr.Textbox(label="Hypotheses", lines=8)

    with gr.Row():
        guidance_box = gr.Textbox(label="Guidance", lines=4)
        kg_box = gr.Textbox(label="KG Hits", lines=6)

    rule_box = gr.Textbox(label="Rule Hits", lines=4)
    cache_box = gr.Textbox(label="Cache Stats", lines=6)
    history_state = gr.State([])

    if USE_STREAM:
        btn.click(
            chat_fn_stream,
            inputs=[inp, history_state],
            outputs=[out, evidence_box, hypothesis_box, guidance_box, kg_box, rule_box, cache_box],
        )
    else:
        btn.click(
            chat_fn,
            inputs=[inp, history_state],
            outputs=[out, evidence_box, hypothesis_box, guidance_box, kg_box, rule_box, cache_box],
        )

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.queue().launch(server_name="0.0.0.0", server_port=port, share=False)
