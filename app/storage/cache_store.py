import json
from typing import Optional

try:
    import redis.asyncio as redis
except Exception:
    redis = None

from app.utils.cache_stats import mark_hit, mark_miss

class CacheStore:
    def __init__(self, url: str, disabled: bool = False, name: str = "cache"):
        # name used for hit/miss stats
        self.disabled = disabled
        self.client = None
        self.fallback = {}
        self.name = name
        if disabled or redis is None:
            return
        try:
            self.client = redis.from_url(url, decode_responses=True)
        except Exception:
            self.client = None

    async def get(self, key: str) -> Optional[str]:
        # Try Redis first; fallback to in-memory dict
        if self.disabled or self.client is None:
            val = self.fallback.get(key)
            if val is None:
                mark_miss(self.name)
            else:
                mark_hit(self.name)
            return val
        try:
            val = await self.client.get(key)
            if val is None:
                mark_miss(self.name)
            else:
                mark_hit(self.name)
            return val
        except Exception:
            mark_miss(self.name)
            return self.fallback.get(key)

    async def set(self, key: str, value: str, ttl: int = 3600):
        if self.disabled or self.client is None:
            self.fallback[key] = value
            return
        try:
            await self.client.setex(key, ttl, value)
        except Exception:
            self.fallback[key] = value
