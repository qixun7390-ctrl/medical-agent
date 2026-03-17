from fastapi import FastAPI
from app.api.routes import router as api_router
from app.observability.metrics import metrics_router


def create_app() -> FastAPI:
    app = FastAPI(title="Medical Agent", version="0.1.0")

    app.include_router(api_router)
    app.include_router(metrics_router)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app
