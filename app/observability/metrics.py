from fastapi import APIRouter
from prometheus_client import Counter, Histogram, generate_latest

router = APIRouter()

REQS = Counter("requests_total", "Total requests")
LAT = Histogram("request_latency_ms", "Latency", buckets=(50,100,200,500,1000,2000,5000))

@router.get("/metrics")
async def metrics():
    return generate_latest()

metrics_router = router
