from app.models.schemas import ChatRequest

class Planner:
    async def create_plan(self, req: ChatRequest) -> dict:
        return {
            "steps": ["retrieve", "rerank", "synthesize"],
            "query": req.query,
        }
