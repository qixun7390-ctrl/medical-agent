import hashlib
from typing import List, Dict
import httpx

from app.core.config import settings


class ExternalSearchClient:
    def __init__(self) -> None:
        self.enabled = settings.search_enabled
        self.provider = settings.search_provider
        self.api_key = settings.search_api_key
        self.allowlist = [d.strip() for d in settings.search_allowlist.split(",") if d.strip()]
        self.timeout = settings.search_timeout

    async def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.enabled:
            return []
        if self.provider == "ncbi_pubmed":
            return await self._search_ncbi(query, top_k)
        return await self._search_serpapi(query, top_k)

    def _build_allowlist_query(self, query: str) -> str:
        if not self.allowlist:
            return query
        site_part = " OR ".join([f"site:{d}" for d in self.allowlist])
        return f"({site_part}) {query}"

    async def _search_serpapi(self, query: str, top_k: int) -> List[Dict]:
        if not self.api_key:
            return []
        q = self._build_allowlist_query(query)
        params = {
            "engine": "google",
            "q": q,
            "num": top_k,
            "api_key": self.api_key,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get("https://serpapi.com/search", params=params)
            r.raise_for_status()
            data = r.json()
        results = []
        for item in data.get("organic_results", [])[:top_k]:
            url = item.get("link") or ""
            title = item.get("title") or ""
            snippet = item.get("snippet") or ""
            doc_id = "web:" + hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
            results.append({
                "doc_id": doc_id,
                "score": 0.6,
                "snippet": f"{title} - {snippet}".strip(" -"),
                "source": url,
            })
        return results

    async def _search_ncbi(self, query: str, top_k: int) -> List[Dict]:
        # PubMed via NCBI E-utilities (ESearch + ESummary)
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": top_k,
            "retmode": "json",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get(f"{base}/esearch.fcgi", params=params)
            r.raise_for_status()
            data = r.json()
            idlist = data.get("esearchresult", {}).get("idlist", [])
            if not idlist:
                return []
            r2 = await client.get(f"{base}/esummary.fcgi", params={
                "db": "pubmed",
                "id": ",".join(idlist),
                "retmode": "json",
                **({"api_key": self.api_key} if self.api_key else {}),
            })
            r2.raise_for_status()
            summ = r2.json().get("result", {})

        results = []
        for pid in idlist:
            item = summ.get(pid, {})
            title = item.get("title", "")
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
            doc_id = "web:" + hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
            results.append({
                "doc_id": doc_id,
                "score": 0.6,
                "snippet": title,
                "source": url,
            })
        return results
