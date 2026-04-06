import httpx
import asyncio
from duckduckgo_search import DDGS


async def web_search(query: str, max_results: int = 4) -> list[dict]:
    """Search the web using DuckDuckGo (free, no API key needed)."""
    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(None, _ddg_search, query, max_results)
        return results
    except Exception:
        return [{"title": "Search unavailable", "url": "", "snippet": "Could not retrieve search results."}]


def _ddg_search(query: str, max_results: int) -> list[dict]:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return [{"title": r["title"], "url": r["href"], "snippet": r["body"]} for r in results]


async def wikipedia_search(query: str) -> tuple[str, str]:
    """Search Wikipedia and return (content, url)."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            search_resp = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": 1,
                },
            )
            search_data = search_resp.json()
            hits = search_data.get("query", {}).get("search", [])
            if not hits:
                return "No Wikipedia article found.", ""

            title = hits[0]["title"]
            summary_resp = await client.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
            )
            if summary_resp.status_code == 200:
                data = summary_resp.json()
                content = f"**{data['title']}**\n\n{data.get('extract', '')}"
                url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
                return content, url
        except Exception:
            pass
    return "Could not fetch Wikipedia article.", ""
