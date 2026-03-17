from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from app.models.schemas import Evidence, Hypothesis

class AgentState(BaseModel):
    """
    重点解析：
    extracted_slots：规则引擎执行后填充的结构化信息
    rule_hints：规则匹配结果（命中规则名称列表）
    guidance：合并的建议文本
    """
    user_id: str
    session_id: str
    query: str
    history: List[Dict[str, str]] = []
    working_memory: Dict[str, Any] = {}
    episodic_summary: Optional[str] = None

    risk_level: str = "unknown"
    extracted_slots: Dict[str, Any] = {}
    rule_hints: List[str] = []
    rule_hits: List[str] = []
    rule_latency_ms: float = 0.0
    rule_evidence: List[Dict[str, Any]] = []
    subqueries: List[str] = []
    candidates: List[Dict[str, Any]] = []
    evidences: List[Evidence] = []
    hypotheses: List[Hypothesis] = []
    guidance: str = ""
    answer: str = ""
