import httpx
import asyncio
import json
from app.core.config import settings

class LLMClient:
    def __init__(self):
        self.endpoint = settings.llm_endpoint
        self.model = settings.llm_model

    async def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        # Simple retry to smooth transient connection errors
        last_err = None
        for _ in range(3):
            try:
                async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
                    r = await client.post(f"{self.endpoint}/chat/completions", json=payload)
                    r.raise_for_status()
                    return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                await asyncio.sleep(0.5)
        raise last_err

    async def generate_stream(self, prompt: str):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{self.endpoint}/chat/completions", json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[len("data: "):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                            delta = obj["choices"][0].get("delta", {}).get("content", "")
                            if delta:
                                yield delta
                        except Exception:
                            continue
