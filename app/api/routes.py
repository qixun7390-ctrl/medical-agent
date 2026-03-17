import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from app.agent.coordinator import AgentCoordinator
from app.models.schemas import ChatRequest, ChatResponse
from app.utils.cache_stats import get_stats
from app.api.rules import get_rules_data, add_rule, render_rules_ui, RuleInput
from app.agent.numeric_parser import parse_numeric_slots
from app.agent.rule_engine import apply_rules
from app.agent import nodes as agent_nodes

router = APIRouter(prefix="/api")
coordinator = AgentCoordinator()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    resp = await coordinator.run(req)
    return JSONResponse(content=resp.model_dump(), media_type="application/json; charset=utf-8")


@router.post("/chat_stream")
async def chat_stream(req: ChatRequest):
    async def event_gen():
        state = await coordinator.prepare_state(req)
        evidence_payload = [e.model_dump() for e in state.evidences]
        rule_hits = state.rule_hits or []
        yield f"event: evidence\ndata: {json.dumps(evidence_payload, ensure_ascii=False)}\n\n"
        yield f"event: rule_hits\ndata: {json.dumps(rule_hits, ensure_ascii=False)}\n\n"

        try:
            async for chunk in coordinator.run_stream_from_state(state):
                yield f"event: answer_chunk\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps(str(e), ensure_ascii=False)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.post("/debug_rules")
async def debug_rules(req: ChatRequest):
    slots = parse_numeric_slots(req.query)
    rules = apply_rules(req.query, slots, agent_nodes.RULES_PATH)
    return JSONResponse(content={
        "query": req.query,
        "slots": slots,
        "rules_path": agent_nodes.RULES_PATH,
        "rule_hits": rules.get("hits", []),
        "rule_evidence": rules.get("evidence", []),
        "rule_add_queries": rules.get("add_queries", []),
        "rule_guidance": rules.get("guidance", []),
    }, media_type="application/json; charset=utf-8")


@router.get("/cache_stats")
async def cache_stats():
    return get_stats()


@router.get("/rules")
async def rules_get():
    return get_rules_data()


@router.post("/rules")
async def rules_post(payload: RuleInput):
    try:
        return add_rule(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/rules_ui")
async def rules_ui():
    return render_rules_ui()
