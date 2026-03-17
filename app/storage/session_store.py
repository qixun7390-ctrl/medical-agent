import os
import json
import time
from app.core.config import settings

try:
    import redis.asyncio as redis
except Exception:
    redis = None

class SessionStore:
    def __init__(self):
        self.client = None
        self.fallback = {}
        if os.environ.get("REDIS_DISABLED", "0") == "1":
            return
        if redis is not None:
            try:
                self.client = redis.from_url(settings.redis_url, decode_responses=True)
            except Exception:
                self.client = None

    async def get_session(self, user_id: str, session_id: str):
        key = f"sess:{user_id}:{session_id}"
        if self.client is None:
            return self.fallback.get(key)
        try:
            raw = await self.client.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            return self.fallback.get(key)

    async def save_session(self, user_id: str, session_id: str, history: list, working_memory: dict):
        key = f"sess:{user_id}:{session_id}"
        payload = {
            "history": history[-10:],
            "working_memory": working_memory,
            "last_updated": int(time.time())
        }
        if self.client is None:
            self.fallback[key] = payload
            return
        try:
            await self.client.setex(key, 3600, json.dumps(payload, ensure_ascii=False))
        except Exception:
            self.fallback[key] = payload
