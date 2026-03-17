from pathlib import Path
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parents[2]


def _default_data_path(rel: str) -> str:
    # Prefer sibling medical-agent data if present, otherwise use project root data
    candidate = BASE_DIR / "medical-agent" / rel
    if candidate.exists():
        return str(candidate)
    return str(BASE_DIR / rel)


class Settings(BaseSettings):
    app_name: str = "medical-agent"
    env: str = "dev"

    # LLM
    llm_endpoint: str = "http://localhost:8001/v1"
    llm_model: str = "/models/Qwen2.5-1.5B-Instruct"
    llm_timeout: float = 8.0

    # Embeddings
    embedding_model: str = "E:/Models/bge-small-zh-v1.5"

    # Reranker
    reranker_model: str = "E:/Models/bge-reranker-v2-m3"

    # Vector store
    vector_index_path: str = _default_data_path("data/processed/faiss_qa.index")
    vector_meta_path: str = _default_data_path("data/processed/meta_qa.jsonl")

    # Retrieval mode
    retrieval_mode: str = "qa"

    # Cache (for hot questions)
    cache_enabled: bool = False
    cache_ttl_seconds: int = 3600

    # KG
    kg_path: str = _default_data_path("data/processed/kg.graph.json")

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Postgres
    db_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/medical_agent"

    # External search (optional)
    search_enabled: bool = False
    search_provider: str = "serpapi"  # serpapi | ncbi_pubmed
    search_api_key: str = ""
    search_allowlist: str = "who.int,cdc.gov,nih.gov,ncbi.nlm.nih.gov"
    search_timeout: float = 8.0

    # Context engineering budgets (approx tokens)
    # 系统指令只需要40，-200方便未来拓展以及算上标签和vllm的tokenizer实际的
    context_system_budget: int = 200

    # 历史对话：200-500 tokens
    context_history_budget: int = 800

    # 检索证据：300-500 tokens
    context_evidence_budget: int = 1000

    # 思维链和输出格式设预算-占 1170 tokens 固定开销

    # 总输入预算应该是：固定部分(1170) + 历史(800) + 证据(1000) + 规则(200) + 问题(50) = 3220
    context_input_budget: int = 3500



settings = Settings()
