from app.rag.pipeline import RagPipeline
from app.models.schemas import ChatRequest

class Executor:
    def __init__(self):
        self.rag = RagPipeline()

    async def execute(self, plan: dict, req: ChatRequest) -> dict:
        return await self.rag.run(req)
