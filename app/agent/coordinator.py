from app.agent.graph import build_graph
from app.agent.state import AgentState
from app.models.schemas import ChatRequest, ChatResponse
from app.agent import nodes
from app.rag.generator import Generator

class AgentCoordinator:
    def __init__(self):
        self.graph = build_graph()
        self.generator = Generator()

    async def run(self, req: ChatRequest) -> ChatResponse:
        state = {
            "user_id": req.user_id,
            "session_id": req.session_id,
            "query": req.query,
            "history": [m.model_dump() for m in req.history],
        }
        result = await self.graph.ainvoke(state)
        return ChatResponse(
            answer=result["answer"],
            evidences=result["evidences"],
            hypotheses=result["hypotheses"],
            guidance=result["guidance"],
            latency_ms=result.get("latency_ms", 0.0),
            model=result.get("model", ""),
            rule_hits=result.get("rule_hits", [])
        )

    async def prepare_state(self, req: ChatRequest) -> AgentState:
        state = AgentState(
            user_id=req.user_id,
            session_id=req.session_id,
            query=req.query,
            history=[m.model_dump() for m in req.history],
        )
        state = await nodes.load_session(state)
        state = await nodes.risk_classifier(state)
        if state.risk_level == "high":
            return state
        state = await nodes.query_expansion(state)
        state = await nodes.multi_retrieval(state)
        state = await nodes.rerank(state)
        state = await nodes.hypothesis_tree(state)
        return state

    async def run_stream_from_state(self, state: AgentState):
        if state.risk_level == "high":
            yield "检测到高风险症状，请立即就医。"
            state.answer = "检测到高风险症状，请立即就医。"
            await nodes.save_session(state)
            return
        if not state.evidences:
            state.answer = "证据不足，无法给出可靠结论。建议就医或提供更多信息。"
            yield state.answer
            await nodes.save_session(state)
            return

        chunks = []
        async for ch in self.generator.stream_from_state(state):
            chunks.append(ch)
            yield ch
        state.answer = "".join(chunks).strip()
        await nodes.save_session(state)
