import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not set in the environment (.env).")
        _client = TavilyClient(api_key=api_key)
    return _client


def web_verify(query, max_results=5):
    """
    Runs a real-time web search via Tavily to fact-check a specific claim
    or figure the user asserted (e.g. "India repo rate October 2026").
    Returns a list of {"source", "text", "url"} dicts. Never raises - on
    any failure it logs a warning and returns an empty list, so the
    calling node degrades the same way news_tool.py does on a dead feed.
    """
    try:
        client = _get_client()
        response = client.search(query, search_depth="basic", max_results=max_results)
        results = []
        for r in response.get("results", []):
            results.append({
                "source": r.get("title", r.get("url", "web result")),
                "text": r.get("content", ""),
                "url": r.get("url", ""),
            })
        return results
    except Exception as e:
        print(f"[verifier] Warning: web search failed: {e}")
        return []


if __name__ == "__main__":
    results = web_verify("India repo rate October 2026 RBI")
    print(f"Fetched {len(results)} results.\n")
    for r in results:
        print(f"[{r['source']}] {r['text'][:200]}")
        print(f"  {r['url']}")
        print()