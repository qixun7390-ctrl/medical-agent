import os
import time
import re
from typing import List
from app.agent.state import AgentState
from app.agent.numeric_parser import parse_numeric_slots
from app.agent.rule_engine import apply_rules
from app.core.config import settings
from app.storage.session_store import SessionStore
from app.storage.memory_store import MemoryStore
from app.rag.retriever import Retriever
from app.rag.generator import Generator
from app.rag.external_search import ExternalSearchClient
from app.models.schemas import Evidence, Hypothesis

# Initialize shared components once.
_session = SessionStore()
_memory = MemoryStore()
_retriever = Retriever()
_generator = Generator()
_external_search = ExternalSearchClient()

# Reranker is optional and can be disabled during eval.
_use_reranker = os.environ.get("RERANKER_DISABLED", "0") != "1"
_reranker = None
if _use_reranker:
    from app.rag.reranker import Reranker
    _reranker = Reranker()

MIN_EVIDENCE_SCORE = float(os.environ.get("MIN_EVIDENCE_SCORE", "0.2"))
RULES_PATH = os.environ.get(
    "CLINICAL_RULES_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "medical-agent", "configs", "clinical_rules.json"),
)

HIGH_RISK_KEYWORDS = [
    "胸痛", "意识障碍", "呼吸困难", "抽搐", "昏迷", "剧烈头痛", "大出血", "呕血"
]

# Step 1: load session + memory
async def load_session(state: AgentState) -> AgentState:
    sess = await _session.get_session(state.user_id, state.session_id)
    if sess:
        state.history = sess.get("history", [])
        state.working_memory = sess.get("working_memory", {})
    state.episodic_summary = await _memory.get_summary(state.user_id)
    return state

# Step 2: risk gating (high-risk -> immediate guidance)
async def risk_classifier(state: AgentState) -> AgentState:
    text = state.query
    if any(k in text for k in HIGH_RISK_KEYWORDS):
        state.risk_level = "high"
        state.guidance = "检测到高风险症状，建议立即就医或拨打急救电话。"
    else:
        state.risk_level = "normal"
    return state

# Step 3: query expansion (simple version)
async def query_expansion(state: AgentState) -> AgentState:
    # Base query
    state.subqueries = [state.query]

    # Normalize temperature queries to reduce numeric drift (38 vs 40)
    def _normalize_query(q: str) -> str:
        return re.sub(
            r"(发烧|发热|高烧|低烧)?\s*(\d{2}(?:\.\d)?)\s*(?:度|℃|c|C)\b",
            lambda m: f"发热 体温 {m.group(2)}℃",
            q,
        )

    norm_q = _normalize_query(state.query)
    if norm_q != state.query:
        state.subqueries.append(norm_q)

    # Parse numeric slots and apply clinical rules
    start = time.perf_counter()
    state.extracted_slots = parse_numeric_slots(state.query)
    rule_out = apply_rules(state.query, state.extracted_slots, RULES_PATH)
    state.rule_hints = rule_out.get("add_queries", [])
    state.rule_hits = rule_out.get("hits", [])
    state.rule_evidence = rule_out.get("evidence", [])
    state.subqueries.extend(state.rule_hints)
    # Merge guidance hints if any
    if rule_out.get("guidance"):
        state.guidance = " ".join(list(dict.fromkeys([state.guidance] + rule_out["guidance"]))).strip()
    state.rule_latency_ms = (time.perf_counter() - start) * 1000
    return state

# Step 4: retrieval
async def multi_retrieval(state: AgentState) -> AgentState:
    candidates = []
    for q in state.subqueries:
        hits = await _retriever.retrieve(q)
        candidates.extend(hits)
    state.candidates = candidates
    return state

# Step 5: rerank + evidence filtering
async def rerank(state: AgentState) -> AgentState:
    if _reranker is None:
        reranked = sorted(state.candidates, key=lambda x: x.get("score", 0), reverse=True)[:12]
    else:
        reranked = await _reranker.rerank(state.query, state.candidates)

    # Filter by score and deduplicate by doc_id
    seen = set()
    filtered = []
    for e in reranked:
        if e.get("score", 0) < MIN_EVIDENCE_SCORE:
            continue
        doc_id = e.get("doc_id")
        if doc_id in seen:
            continue
        seen.add(doc_id)
        filtered.append(e)

    state.evidences = [Evidence(**e) for e in filtered[:8]]

    # Prepend rule evidence (if any) to prioritize rule-based differentiation
    seen_ids = {e.doc_id for e in state.evidences}
    rule_evs = []
    for e in state.rule_evidence:
        try:
            if e.get("doc_id") in seen_ids:
                continue
            rule_evs.append(Evidence(**e))
            seen_ids.add(e.get("doc_id"))
        except Exception:
            continue
    if rule_evs:
        state.evidences = rule_evs + state.evidences

    # If evidence is weak, optionally augment with external search
    if settings.search_enabled and len(state.evidences) < 3:
        ext_hits = await _external_search.search(state.query, top_k=3)
        for e in ext_hits:
            try:
                state.evidences.append(Evidence(**e))
            except Exception:
                continue
    return state

# Step 6: hypothesis tree
async def hypothesis_tree(state: AgentState) -> AgentState:
    hyps: List[Hypothesis] = []
    for i, ev in enumerate(state.evidences[:3]):
        hyps.append(Hypothesis(
            title=f"可能情况 {i+1}",
            confidence="中",
            evidence_ids=[ev.doc_id],
        ))
    state.hypotheses = hyps
    return state

# Step 7: generation (with refusal if no evidence)
async def generate_answer(state: AgentState) -> AgentState:
    start = time.perf_counter()
    if state.risk_level == "high":
        state.answer = "检测到高风险症状，请立即就医。"
    elif not state.evidences:
        state.answer = "证据不足，无法给出可靠结论。建议就医或提供更多信息。"
    else:
        result = await _generator.generate_from_state(state)
        state.answer = result["answer"]
        state.guidance = result.get("guidance", state.guidance)
    state.working_memory["last_query"] = state.query
    state.working_memory["last_answer"] = state.answer
    state.working_memory["last_evidence_ids"] = [e.doc_id for e in state.evidences]
    state.working_memory["latency_ms"] = (time.perf_counter() - start) * 1000
    return state

# Step 8: persist session
async def save_session(state: AgentState) -> AgentState:
    # append current turn into history
    if state.query:
        state.history.append({"role": "user", "content": state.query})
    if state.answer:
        state.history.append({"role": "assistant", "content": state.answer})
    await _session.save_session(state.user_id, state.session_id, state.history, state.working_memory)
    await _memory.maybe_write_summary(state.user_id, state.history, state.answer)
    return state
