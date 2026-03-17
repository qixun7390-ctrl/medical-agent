from app.llm.client import LLMClient
from app.agent.state import AgentState
from app.storage.cache_store import CacheStore
from app.core.config import settings
from app.rag.context_engineer import ContextEngineer

class Generator:
    def __init__(self):
        self.llm = LLMClient()
        self.cache = CacheStore(settings.redis_url, disabled=not settings.cache_enabled, name="answer")
        self.ctx = ContextEngineer()

    async def generate_from_state(self, state: AgentState):
        # 1) Answer cache for hot queries
        cache_key = f"answer:{state.query}"
        cached = await self.cache.get(cache_key)
        if cached:
            return {"answer": cached, "guidance": "(cached) 如症状持续或加重，请及时就医。", "model": self.llm.model}

        # 2) First pass generation
        try:
            prompt = self._build_prompt(state, force_citations=False)
            answer = await self.llm.generate(prompt)
        except Exception:
            # If LLM service is unavailable, return evidence-only guidance
            answer = self._rule_only_answer(state)
            guidance = "如症状持续或加重，请及时就医。"
            return {"answer": answer, "guidance": guidance, "model": self.llm.model}

        # 3) If no citations, force a stricter second pass
        if not self._has_citations(answer):
            prompt2 = self._build_prompt(state, force_citations=True)
            answer = await self.llm.generate(prompt2)

        # 4) Still no citations -> refuse
        if not self._has_citations(answer):
            answer = self._rule_only_answer(state)

        await self.cache.set(cache_key, answer, ttl=settings.cache_ttl_seconds)
        guidance = "如症状持续或加重，请及时就医。"
        return {"answer": answer, "guidance": guidance, "model": self.llm.model}

    def _has_citations(self, text: str) -> bool:
        return "[" in text and "]" in text

    def _build_prompt(self, state: AgentState, force_citations: bool = False) -> str:
        evidences = [f"[{e.doc_id}] {e.snippet}" for e in state.evidences]
        return self.ctx.build_prompt(state, evidences, force_citations=force_citations)

    def _rule_only_answer(self, state: AgentState) -> str:
        # Deterministic fallback using rule evidence if present
        if not state.evidences:
            return "证据不足，无法给出可靠结论。建议就医或提供更多信息。"
        # Build a concise natural-language response with citations (no raw snippets)
        top = state.evidences[:3]
        summary = "根据现有证据，建议尽快就医评估，并根据症状严重程度选择门诊或急诊。"
        cites = " ".join([f"[{e.doc_id}]" for e in top if e.doc_id])
        return f"{summary} {cites}\n如症状持续或加重，请及时就医。"

    async def stream_from_state(self, state: AgentState):
        # Streaming generation with strict citations
        cache_key = f"answer:{state.query}"
        cached = await self.cache.get(cache_key)
        if cached:
            yield cached
            return

        prompt = self._build_prompt(state, force_citations=True)
        chunks = []
        async for chunk in self.llm.generate_stream(prompt):
            chunks.append(chunk)
            yield chunk

        answer = "".join(chunks).strip()
        if not answer or not self._has_citations(answer):
            answer = self._rule_only_answer(state)
        await self.cache.set(cache_key, answer, ttl=settings.cache_ttl_seconds)
