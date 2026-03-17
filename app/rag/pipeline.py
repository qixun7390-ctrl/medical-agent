from app.rag.retriever import Retriever
from app.rag.reranker import Reranker
from app.rag.generator import Generator
from app.models.schemas import ChatRequest

class RagPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.generator = Generator()

    async def run(self, req: ChatRequest) -> dict:
        candidates = await self.retriever.retrieve(req.query)
        reranked = await self.reranker.rerank(req.query, candidates)
        answer = await self.generator.generate(req, reranked)
        return answer
