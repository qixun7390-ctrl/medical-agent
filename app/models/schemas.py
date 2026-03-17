from typing import List, Optional
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    query: str
    history: List[Message] = []
    images: Optional[List[str]] = None

class Evidence(BaseModel):
    doc_id: str
    score: float
    snippet: str
    source: Optional[str] = None

class Hypothesis(BaseModel):
    title: str
    confidence: str
    evidence_ids: List[str]

class ChatResponse(BaseModel):
    answer: str
    evidences: List[Evidence]
    hypotheses: List[Hypothesis]
    guidance: str
    latency_ms: float
    model: str
    rule_hits: Optional[List[str]] = None
