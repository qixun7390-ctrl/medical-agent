from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.agent import nodes

# Build the LangGraph state machine for the agent.
# This is the top-level reasoning pipeline.

def build_graph():
    g = StateGraph(AgentState)

    # 1) Load session + memory
    g.add_node("load_session", nodes.load_session)
    # 2) Risk gating (high-risk -> immediate guidance)
    g.add_node("risk", nodes.risk_classifier)
    # 3) Query expansion
    g.add_node("expand", nodes.query_expansion)
    # 4) Retrieval (vector/KG)
    g.add_node("retrieve", nodes.multi_retrieval)
    # 5) Rerank (optional)
    g.add_node("rerank", nodes.rerank)
    # 6) Hypothesis tree
    g.add_node("hypothesis", nodes.hypothesis_tree)
    # 7) Generation
    g.add_node("generate", nodes.generate_answer)
    # 8) Persist
    g.add_node("save_session", nodes.save_session)

    # Entry
    g.set_entry_point("load_session")
    g.add_edge("load_session", "risk")

    # Route high-risk directly to generation
    def route_after_risk(state: AgentState):
        if state.risk_level == "high":
            return "generate"
        return "expand"

    g.add_conditional_edges("risk", route_after_risk, {
        "generate": "generate",
        "expand": "expand",
    })

    # Normal path
    g.add_edge("expand", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "hypothesis")
    g.add_edge("hypothesis", "generate")
    g.add_edge("generate", "save_session")
    g.add_edge("save_session", END)

    return g.compile()
