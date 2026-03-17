import os
from typing import Optional
from app.core.config import settings


class MemoryStore:
    def __init__(self):
        self.engine = None
        self._text = None
        if os.environ.get("PG_DISABLED", "0") == "1":
            return
        try:
            from sqlalchemy import text
            from sqlalchemy.ext.asyncio import create_async_engine
            self._text = text
            self.engine = create_async_engine(settings.db_url, echo=False, future=True)
        except Exception:
            self.engine = None

    async def _ensure_tables(self):
        if not self.engine:
            return False
        async with self.engine.begin() as conn:
            await conn.execute(self._text("""
                CREATE TABLE IF NOT EXISTS session_summaries (
                    user_id TEXT PRIMARY KEY,
                    summary TEXT,
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """))
        return True

    async def get_summary(self, user_id: str) -> Optional[str]:
        ok = await self._ensure_tables()
        if not ok:
            return None
        try:
            async with self.engine.begin() as conn:
                res = await conn.execute(self._text("SELECT summary FROM session_summaries WHERE user_id=:uid"), {"uid": user_id})
                row = res.fetchone()
                return row[0] if row else None
        except Exception:
            return None

    async def maybe_write_summary(self, user_id: str, history: list, answer: str):
        ok = await self._ensure_tables()
        if not ok:
            return None
        try:
            last_turns = history[-4:] if history else []
            lines = []
            for m in last_turns:
                role = m.get("role", "")
                content = m.get("content", "")
                if content:
                    lines.append(f"{role}: {content}")
            lines.append(f"assistant: {answer}")
            summary = " | ".join(lines)[:1000]

            async with self.engine.begin() as conn:
                await conn.execute(self._text("""
                    INSERT INTO session_summaries (user_id, summary, updated_at)
                    VALUES (:uid, :summary, NOW())
                    ON CONFLICT (user_id)
                    DO UPDATE SET summary = EXCLUDED.summary, updated_at = NOW();
                """), {"uid": user_id, "summary": summary})
        except Exception:
            return None

    async def add_event(self, user_id: str, key: str, value: str):
        ok = await self._ensure_tables()
        if not ok:
            return None
        try:
            async with self.engine.begin() as conn:
                await conn.execute(self._text("""
                    CREATE TABLE IF NOT EXISTS user_events (
                        user_id TEXT,
                        key TEXT,
                        value TEXT,
                        ts TIMESTAMP DEFAULT NOW()
                    );
                """))
                await conn.execute(self._text("""
                    INSERT INTO user_events (user_id, key, value, ts)
                    VALUES (:uid, :key, :value, NOW());
                """), {"uid": user_id, "key": key, "value": value})
        except Exception:
            return None
