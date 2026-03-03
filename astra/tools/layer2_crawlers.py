"""
ASTRA Layer 2 — Multi-Source Crawlers
Action paradigm: HYBRID (JSON for single calls, code for aggregation)

Tools:
  duckduckgo_search, jina_fetch_url, firecrawl_scrape, firecrawl_crawl,
  arxiv_search, semantic_scholar_search, openalex_search,
  pubmed_search, github_search, searxng_search
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Optional
from urllib.parse import quote

import httpx
import requests
from langchain_core.tools import tool
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from astra.config import get_config

# ─── Tavily ──────────────────────────────────────────────────────────────────

@tool
def tavily_search(
    query: str,
    max_results: int = 10,
    search_depth: str = "advanced",
    include_raw_content: bool = False,
) -> list[dict]:
    """
    Layer 2: Web search via Tavily AI. Superior quality results with full-page content.
    Requires TAVILY_API_KEY. Falls back to DuckDuckGo when key is absent.

    Args:
        query: Search query string.
        max_results: Maximum results to return (default 10).
        search_depth: "basic" or "advanced" (advanced gets full content).
        include_raw_content: Include raw HTML page content.

    Returns:
        List of {title, url, snippet, raw_content, source_type} dicts.
    """
    logger.info(f"[Layer 2] Tavily: '{query[:60]}' max={max_results}")
    cfg = get_config()

    if not cfg.tavily_api_key:
        logger.warning("[Layer 2] TAVILY_API_KEY not set — falling back to DuckDuckGo")
        return duckduckgo_search.invoke(
            {"query": query, "max_results": max_results}
        )

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=cfg.tavily_api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_raw_content=include_raw_content,
        )

        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
                "raw_content": r.get("raw_content", "") if include_raw_content else "",
                "score": r.get("score", 0.0),
                "source_type": "web",
            })

        logger.info(f"[Layer 2] Tavily returned {len(results)} results")
        return results

    except ImportError:
        logger.warning("[Layer 2] tavily-python not installed — falling back to DuckDuckGo")
        return duckduckgo_search.invoke(
            {"query": query, "max_results": max_results}
        )
    except Exception as e:
        logger.error(f"[Layer 2] Tavily failed: {e} — falling back to DuckDuckGo")
        return duckduckgo_search.invoke(
            {"query": query, "max_results": max_results}
        )


# ─── DuckDuckGo ─────────────────────────────────────────────────────────────

@tool
def duckduckgo_search(
    query: str,
    max_results: int = 10,
    region: str = "wt-wt",
    time_filter: Optional[str] = None,
) -> list[dict]:
    """
    Layer 2: Web search via DuckDuckGo. Free, no API key required.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (default 10).
        region: Region code (default "wt-wt" = worldwide).
        time_filter: "d" (day), "w" (week), "m" (month), or None.

    Returns:
        List of {title, url, snippet} dicts.
    """
    logger.info(f"[Layer 2] DuckDuckGo: '{query[:60]}' max={max_results}")
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            # ddgs (new) uses 'query'; duckduckgo_search (old) uses 'keywords'
            try:
                raw = list(ddgs.text(query, region=region, max_results=max_results,
                                     timelimit=time_filter))
            except TypeError:
                raw = list(ddgs.text(keywords=query, region=region,
                                     max_results=max_results,
                                     timelimit=time_filter))
            for r in raw:
                results.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                        "source_type": "web",
                    }
                )
                if len(results) >= max_results:
                    break

        logger.info(f"[Layer 2] DuckDuckGo returned {len(results)} results")
        return results

    except Exception as e:
        logger.error(f"[Layer 2] DuckDuckGo failed: {e}")
        return []


# ─── SearXNG Fallback ────────────────────────────────────────────────────────

@tool
def searxng_search(query: str, max_results: int = 20) -> list[dict]:
    """
    Layer 2: Self-hosted SearXNG meta-search fallback.
    Used when DuckDuckGo is rate-limited.

    Args:
        query: Search query.
        max_results: Max results.

    Returns:
        List of {title, url, snippet} dicts.
    """
    cfg = get_config()
    if not cfg.astra_searxng_enabled:
        logger.warning("[Layer 2] SearXNG disabled, returning empty")
        return []

    try:
        resp = requests.get(
            f"{cfg.astra_searxng_base_url}/search",
            params={"q": query, "format": "json", "engines": "google,bing,duckduckgo"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
                "source_type": "web",
            }
            for r in data.get("results", [])[:max_results]
        ]
        logger.info(f"[Layer 2] SearXNG returned {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"[Layer 2] SearXNG failed: {e}")
        return []


# ─── Jina URL Fetcher ────────────────────────────────────────────────────────

@tool
def jina_fetch_url(url: str, timeout_seconds: int = 30) -> dict:
    """
    Layer 2: Convert any URL to clean markdown via Jina Reader.
    Free, no auth required.

    Args:
        url: URL to fetch.
        timeout_seconds: Request timeout.

    Returns:
        {url, markdown, title} dict.
    """
    cfg = get_config()
    jina_url = f"{cfg.astra_jina_base_url}/{url}"
    logger.info(f"[Layer 2] Jina fetch: {url[:80]}")

    try:
        resp = httpx.get(
            jina_url,
            timeout=timeout_seconds,
            headers={"Accept": "text/plain", "X-Return-Format": "markdown"},
            follow_redirects=True,
        )
        resp.raise_for_status()
        content = resp.text
        # Extract title from first H1 if present
        title = url
        for line in content.split("\n")[:5]:
            if line.startswith("# "):
                title = line[2:].strip()
                break

        return {"url": url, "markdown": content, "title": title}

    except Exception as e:
        logger.error(f"[Layer 2] Jina fetch failed for {url}: {e}")
        return {"url": url, "markdown": "", "title": url}


# ─── Firecrawl ───────────────────────────────────────────────────────────────

@tool
def firecrawl_scrape(
    url: str,
    formats: list = None,
    timeout_ms: int = 30000,
) -> dict:
    """
    Layer 2: Scrape a JS-heavy page via self-hosted Firecrawl.

    Args:
        url: URL to scrape.
        formats: List of output formats, e.g. ["markdown"] (default).
        timeout_ms: Timeout in milliseconds.

    Returns:
        {markdown, html, screenshot, metadata} dict.
    """
    cfg = get_config()
    if not cfg.astra_firecrawl_enabled:
        logger.warning("[Layer 2] Firecrawl disabled, falling back to Jina")
        return jina_fetch_url.invoke({"url": url})  # type: ignore[return-value]

    formats = formats or ["markdown"]
    logger.info(f"[Layer 2] Firecrawl scrape: {url[:80]}")

    try:
        resp = requests.post(
            f"{cfg.astra_firecrawl_base_url}/v1/scrape",
            json={"url": url, "formats": formats, "timeout": timeout_ms},
            headers={
                "Authorization": f"Bearer {cfg.astra_firecrawl_api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout_ms / 1000 + 5,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})
        return {
            "markdown": data.get("markdown", ""),
            "html": data.get("html"),
            "screenshot": data.get("screenshot"),
            "metadata": data.get("metadata", {}),
        }
    except Exception as e:
        logger.error(f"[Layer 2] Firecrawl scrape failed: {e} — falling back to Jina")
        result = jina_fetch_url.invoke({"url": url})
        return {
            "markdown": result.get("markdown", ""),
            "html": None,
            "screenshot": None,
            "metadata": {"title": result.get("title", "")},
        }


@tool
def firecrawl_crawl(
    url: str,
    max_depth: int = 3,
    max_pages: int = 50,
    formats: list = None,
) -> list[dict]:
    """
    Layer 2: Crawl an entire site up to a depth limit (for documentation).

    Args:
        url: Root URL to crawl.
        max_depth: Maximum crawl depth.
        max_pages: Maximum pages to fetch.
        formats: Output formats.

    Returns:
        List of {url, markdown, metadata} dicts.
    """
    cfg = get_config()
    if not cfg.astra_firecrawl_enabled:
        logger.warning("[Layer 2] Firecrawl disabled — returning single page")
        page = firecrawl_scrape.invoke({"url": url})
        return [{"url": url, "markdown": page.get("markdown", ""), "metadata": {}}]

    formats = formats or ["markdown"]
    logger.info(f"[Layer 2] Firecrawl crawl: {url} depth={max_depth} pages={max_pages}")

    try:
        # Start crawl job
        resp = requests.post(
            f"{cfg.astra_firecrawl_base_url}/v1/crawl",
            json={
                "url": url,
                "maxDepth": max_depth,
                "limit": max_pages,
                "scrapeOptions": {"formats": formats},
            },
            headers={
                "Authorization": f"Bearer {cfg.astra_firecrawl_api_key}",
                "Content-Type": "application/json",
            },
            timeout=60,
        )
        resp.raise_for_status()
        job_id = resp.json().get("id")

        if not job_id:
            raise ValueError("No job ID returned from Firecrawl")

        # Poll for completion
        for _ in range(120):  # max ~2 min poll
            time.sleep(1)
            status_resp = requests.get(
                f"{cfg.astra_firecrawl_base_url}/v1/crawl/{job_id}",
                headers={"Authorization": f"Bearer {cfg.astra_firecrawl_api_key}"},
                timeout=10,
            )
            status_data = status_resp.json()
            if status_data.get("status") in ("completed", "failed"):
                break

        pages = status_data.get("data", [])
        results = [
            {
                "url": p.get("metadata", {}).get("sourceURL", url),
                "markdown": p.get("markdown", ""),
                "metadata": p.get("metadata", {}),
            }
            for p in pages
        ]
        logger.info(f"[Layer 2] Firecrawl crawl completed: {len(results)} pages")
        return results

    except Exception as e:
        logger.error(f"[Layer 2] Firecrawl crawl failed: {e}")
        return []


# ─── arXiv ──────────────────────────────────────────────────────────────────

@tool
def arxiv_search(
    query: str,
    max_results: int = 20,
    sort_by: str = "submittedDate",
    date_from: Optional[str] = None,
    categories: Optional[list[str]] = None,
) -> list[dict]:
    """
    Layer 2: Search arXiv. Free, no authentication required.

    Args:
        query: Search query.
        max_results: Max results (default 20).
        sort_by: "submittedDate" | "relevance" | "lastUpdatedDate".
        date_from: Filter papers submitted after this date (YYYY-MM-DD).
        categories: arXiv category filters e.g. ["cs.AI", "cs.CL"].

    Returns:
        List of {arxiv_id, title, authors, abstract, published, pdf_url, categories}.
    """
    logger.info(f"[Layer 2] arXiv search: '{query[:60]}' max={max_results}")

    try:
        import arxiv as arxiv_lib

        search_query = query
        if categories:
            cat_filter = " OR ".join(f"cat:{c}" for c in categories)
            search_query = f"({query}) AND ({cat_filter})"

        sort_map = {
            "submittedDate": arxiv_lib.SortCriterion.SubmittedDate,
            "relevance": arxiv_lib.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv_lib.SortCriterion.LastUpdatedDate,
        }
        criterion = sort_map.get(sort_by, arxiv_lib.SortCriterion.Relevance)

        client = arxiv_lib.Client()
        search = arxiv_lib.Search(
            query=search_query,
            max_results=max_results,
            sort_by=criterion,
        )

        results = []
        for paper in client.results(search):
            published_str = paper.published.strftime("%Y-%m-%d") if paper.published else ""
            if date_from and published_str < date_from:
                continue
            results.append(
                {
                    "arxiv_id": paper.entry_id.split("/")[-1],
                    "title": paper.title,
                    "authors": [a.name for a in paper.authors],
                    "abstract": paper.summary,
                    "published": published_str,
                    "pdf_url": paper.pdf_url or "",
                    "categories": paper.categories,
                    "source_type": "arxiv",
                    "url": paper.entry_id,
                }
            )

        logger.info(f"[Layer 2] arXiv returned {len(results)} papers")
        return results

    except Exception as e:
        logger.error(f"[Layer 2] arXiv search failed: {e}")
        return []


# ─── Semantic Scholar ────────────────────────────────────────────────────────

@tool
def semantic_scholar_search(
    query: str,
    max_results: int = 20,
    fields: Optional[list[str]] = None,
    year_filter: Optional[str] = None,
) -> list[dict]:
    """
    Layer 2: Search Semantic Scholar API. Free, no key for basic use.

    Args:
        query: Search query.
        max_results: Max results.
        fields: Fields to return (default: title, abstract, authors, year, etc.)
        year_filter: e.g. "2024-2026".

    Returns:
        List of {paper_id, title, abstract, year, citation_count, pdf_url, authors}.
    """
    cfg = get_config()
    fields = fields or [
        "title", "abstract", "authors", "year", "citationCount",
        "externalIds", "openAccessPdf",
    ]
    logger.info(f"[Layer 2] Semantic Scholar: '{query[:60]}' max={max_results}")

    try:
        params: dict[str, Any] = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": ",".join(fields),
        }
        if year_filter:
            params["year"] = year_filter

        headers = {}
        if cfg.astra_s2_api_key:
            headers["x-api-key"] = cfg.astra_s2_api_key

        resp = requests.get(
            f"{cfg.astra_s2_base_url}/paper/search",
            params=params,
            headers=headers,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for paper in data.get("data", []):
            pdf = paper.get("openAccessPdf") or {}
            results.append(
                {
                    "paper_id": paper.get("paperId", ""),
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", ""),
                    "year": paper.get("year"),
                    "citation_count": paper.get("citationCount", 0),
                    "pdf_url": pdf.get("url", "") if pdf else "",
                    "authors": [a.get("name", "") for a in paper.get("authors", [])],
                    "source_type": "semantic_scholar",
                    "url": f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}",
                }
            )

        logger.info(f"[Layer 2] Semantic Scholar returned {len(results)} papers")
        return results

    except Exception as e:
        logger.error(f"[Layer 2] Semantic Scholar failed: {e}")
        return []


# ─── OpenAlex ───────────────────────────────────────────────────────────────

@tool
def openalex_search(
    query: str,
    max_results: int = 20,
    filter_str: Optional[str] = None,
    sort: str = "relevance_score:desc",
) -> list[dict]:
    """
    Layer 2: Search OpenAlex (250M+ scholarly works). Fully free, no key.

    Args:
        query: Search query.
        max_results: Max results.
        filter_str: e.g. "publication_year:2024-2026".
        sort: Sort order.

    Returns:
        List of {id, title, abstract, year, cited_by_count, open_access, doi}.
    """
    cfg = get_config()
    logger.info(f"[Layer 2] OpenAlex: '{query[:60]}' max={max_results}")

    try:
        params: dict[str, Any] = {
            "search": query,
            "per-page": min(max_results, 200),
            "sort": sort,
        }
        if filter_str:
            params["filter"] = filter_str
        if cfg.astra_openalex_email:
            params["mailto"] = cfg.astra_openalex_email

        resp = requests.get(
            f"{cfg.astra_openalex_base_url}/works",
            params=params,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for work in data.get("results", []):
            # Extract abstract from inverted index
            abstract_inv = work.get("abstract_inverted_index") or {}
            abstract = _reconstruct_abstract(abstract_inv)

            results.append(
                {
                    "id": work.get("id", ""),
                    "title": work.get("display_name", ""),
                    "abstract": abstract,
                    "year": work.get("publication_year"),
                    "cited_by_count": work.get("cited_by_count", 0),
                    "open_access": work.get("open_access", {}).get("is_oa", False),
                    "doi": work.get("doi", ""),
                    "pdf_url": work.get("open_access", {}).get("oa_url", ""),
                    "source_type": "openalex",
                    "url": work.get("doi", work.get("id", "")),
                }
            )

        logger.info(f"[Layer 2] OpenAlex returned {len(results)} works")
        return results

    except Exception as e:
        logger.error(f"[Layer 2] OpenAlex failed: {e}")
        return []


def _reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions[pos] = word
    return " ".join(positions[i] for i in sorted(positions))


# ─── PubMed ──────────────────────────────────────────────────────────────────

@tool
def pubmed_search(
    query: str,
    max_results: int = 15,
    date_range: Optional[str] = None,
) -> list[dict]:
    """
    Layer 2: Search PubMed via NCBI Entrez API. Free, no key for basic use.

    Args:
        query: Search query.
        max_results: Max results.
        date_range: e.g. "2024/01/01:2026/12/31[dp]".

    Returns:
        List of {pmid, title, abstract, authors, journal, published, pmc_url}.
    """
    cfg = get_config()
    logger.info(f"[Layer 2] PubMed: '{query[:60]}' max={max_results}")

    base = cfg.astra_pubmed_base_url

    try:
        # Search for IDs
        search_params: dict[str, Any] = {
            "db": "pubmed",
            "term": query + (f" AND {date_range}" if date_range else ""),
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        if cfg.astra_pubmed_max_results:
            search_params["api_key"] = ""  # empty key for basic access

        search_resp = requests.get(
            f"{base}/esearch.fcgi", params=search_params, timeout=20
        )
        search_resp.raise_for_status()
        ids = search_resp.json()["esearchresult"]["idlist"]
        if not ids:
            return []

        # Fetch summaries
        fetch_resp = requests.get(
            f"{base}/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            timeout=20,
        )
        fetch_resp.raise_for_status()
        fetch_data = fetch_resp.json().get("result", {})

        results = []
        for pmid in ids:
            item = fetch_data.get(pmid, {})
            if not item or pmid == "uids":
                continue
            pmc_uid = ""
            for art_id in item.get("articleids", []):
                if art_id.get("idtype") == "pmc":
                    pmc_uid = art_id.get("value", "")
            results.append(
                {
                    "pmid": pmid,
                    "title": item.get("title", ""),
                    "abstract": "",  # Requires separate efetch call
                    "authors": [
                        a.get("name", "") for a in item.get("authors", [])
                    ],
                    "journal": item.get("source", ""),
                    "published": item.get("pubdate", ""),
                    "pmc_url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_uid}/"
                    if pmc_uid else "",
                    "source_type": "pubmed",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                }
            )

        logger.info(f"[Layer 2] PubMed returned {len(results)} papers")
        return results

    except Exception as e:
        logger.error(f"[Layer 2] PubMed failed: {e}")
        return []


# ─── GitHub ──────────────────────────────────────────────────────────────────

@tool
def github_search(
    query: str,
    search_type: str = "repositories",
    max_results: int = 10,
    sort: str = "stars",
    language: Optional[str] = None,
) -> list[dict]:
    """
    Layer 2: Search GitHub. Free: 60 req/hr unauthenticated, 5000/hr with token.

    Args:
        query: Search query.
        search_type: "repositories" | "code" | "topics".
        max_results: Max results.
        sort: "stars" | "updated" | "forks".
        language: Filter by programming language.

    Returns:
        List of {full_name, description, stars, url, readme_url, license, topics}.
    """
    cfg = get_config()
    logger.info(f"[Layer 2] GitHub: '{query[:60]}' type={search_type}")

    search_query = query
    if language:
        search_query += f" language:{language}"

    headers: dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
    if cfg.astra_github_token:
        headers["Authorization"] = f"token {cfg.astra_github_token}"

    try:
        resp = requests.get(
            f"https://api.github.com/search/{search_type}",
            params={
                "q": search_query,
                "sort": sort,
                "per_page": min(max_results, 30),
            },
            headers=headers,
            timeout=20,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])

        results = []
        for item in items[:max_results]:
            results.append(
                {
                    "full_name": item.get("full_name", ""),
                    "description": item.get("description", ""),
                    "stars": item.get("stargazers_count", 0),
                    "url": item.get("html_url", ""),
                    "readme_url": f"https://raw.githubusercontent.com/{item.get('full_name', '')}/main/README.md",
                    "license": (item.get("license") or {}).get("spdx_id", ""),
                    "topics": item.get("topics", []),
                    "last_updated": item.get("updated_at", ""),
                    "source_type": "github",
                }
            )

        logger.info(f"[Layer 2] GitHub returned {len(results)} repos")
        return results

    except Exception as e:
        logger.error(f"[Layer 2] GitHub search failed: {e}")
        return []


# ─── Medium ───────────────────────────────────────────────────────────────────

@tool
def medium_fetch_article(url: str) -> dict:
    """
    Layer 2: Fetch a Medium article as clean markdown.

    Fetch chain (in order):
      1. Jina Reader  r.jina.ai/{url}  — handles paywall for most articles
      2. Freedium     freedium.cfd/{url} — free Medium unlocker (fallback)
      3. Direct HTTP  with session cookie if ASTRA_MEDIUM_SESSION_COOKIE is set

    Returns:
        {url, markdown, title, source_type}
    """
    logger.info(f"[Layer 2] Medium fetch: {url[:80]}")
    cfg = get_config()
    empty = {"url": url, "markdown": "", "title": url, "source_type": "medium"}

    # Normalise URL — ensure it's a medium.com URL
    if "medium.com" not in url and "towardsdatascience.com" not in url and "substack" not in url:
        url = "https://medium.com/" + url.lstrip("/")

    # 1. Jina Reader (best — bypasses soft paywall, JS rendering)
    try:
        jina_url = f"{cfg.astra_jina_base_url}/{url}"
        resp = httpx.get(
            jina_url, timeout=30, follow_redirects=True,
            headers={
                "Accept": "text/plain",
                "X-Return-Format": "markdown",
                "X-With-Images-Summary": "true",
            },
        )
        resp.raise_for_status()
        text = resp.text.strip()
        if len(text) > 300:
            title = url
            for line in text.split("\n")[:5]:
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
            logger.info(f"[Layer 2] Medium/Jina: {len(text)} chars from {url[:60]}")
            return {"url": url, "markdown": text, "title": title, "source_type": "medium"}
    except Exception as e:
        logger.debug(f"[Layer 2] Jina failed for Medium {url}: {e}")

    # 2. Freedium (free Medium unlocker)
    try:
        freedium_url = f"https://freedium.cfd/{url}"
        resp = httpx.get(
            freedium_url, timeout=25, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0)"},
        )
        resp.raise_for_status()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        # Freedium wraps article in <main> or <article>
        article = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile(r"article|post|content"))
        if article:
            text = article.get_text("\n", strip=True)
            title_el = soup.find("h1")
            title = title_el.get_text(strip=True) if title_el else url
            if len(text) > 300:
                logger.info(f"[Layer 2] Medium/Freedium: {len(text)} chars from {url[:60]}")
                return {"url": url, "markdown": text, "title": title, "source_type": "medium"}
    except Exception as e:
        logger.debug(f"[Layer 2] Freedium failed for {url}: {e}")

    # 3. Direct fetch with session cookie (subscriber access)
    if cfg.astra_medium_session_cookie:
        try:
            resp = httpx.get(
                url, timeout=20, follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0)",
                    "Cookie": f"sid={cfg.astra_medium_session_cookie}",
                },
            )
            resp.raise_for_status()
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            article = soup.find("article") or soup.find("section")
            if article:
                text = article.get_text("\n", strip=True)
                title_el = soup.find("h1")
                title = title_el.get_text(strip=True) if title_el else url
                if len(text) > 300:
                    logger.info(f"[Layer 2] Medium/direct: {len(text)} chars")
                    return {"url": url, "markdown": text, "title": title, "source_type": "medium"}
        except Exception as e:
            logger.warning(f"[Layer 2] Medium direct fetch failed: {e}")

    return empty


@tool
def medium_search(query: str, max_results: int = 8) -> list[dict]:
    """
    Layer 2: Search Medium/Towards Data Science for articles.

    Uses Tavily/DDG targeted search on medium.com + towardsdatascience.com,
    then enriches each result by fetching the full article.

    Returns:
        List of {title, url, snippet, markdown, source_type} dicts.
    """
    logger.info(f"[Layer 2] Medium search: '{query[:60]}'")

    # Search specifically on Medium domains
    site_query = f'site:medium.com OR site:towardsdatascience.com {query}'
    results: list[dict] = []

    try:
        ddg_results = duckduckgo_search.invoke(
            {"query": site_query, "max_results": max_results}
        )
        for r in ddg_results:
            r["source_type"] = "medium"
            results.append(r)
    except Exception as e:
        logger.warning(f"[Layer 2] Medium search DDG failed: {e}")

    logger.info(f"[Layer 2] Medium search: {len(results)} results for '{query[:40]}'")
    return results


# ─── Substack ─────────────────────────────────────────────────────────────────

@tool
def substack_search(query: str, max_results: int = 8) -> list[dict]:
    """
    Layer 2: Search across Substack newsletters via Substack's discovery API.

    API endpoint: substack.com/api/v1/reader/feed/search
    Falls back to targeted DDG search on substack.com.

    Returns:
        List of {title, url, snippet, authors, source_type} dicts.
    """
    logger.info(f"[Layer 2] Substack search: '{query[:60]}'")
    results: list[dict] = []

    # 1. Substack native search API
    try:
        resp = requests.get(
            "https://substack.com/api/v1/reader/feed/search",
            params={"query": query, "types": "post", "limit": max_results},
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("results", [])[:max_results]:
            post = item.get("post", item)
            canonical = post.get("canonical_url", "")
            if not canonical:
                pub = post.get("publication", {})
                slug = pub.get("subdomain", "")
                post_slug = post.get("slug", "")
                if slug and post_slug:
                    canonical = f"https://{slug}.substack.com/p/{post_slug}"
            results.append({
                "title": post.get("title", ""),
                "url": canonical,
                "snippet": post.get("subtitle", post.get("description", ""))[:400],
                "authors": [post.get("publishedBylines", [{}])[0].get("name", "Unknown")]
                            if post.get("publishedBylines") else ["Unknown"],
                "source_type": "substack",
            })
        if results:
            logger.info(f"[Layer 2] Substack API: {len(results)} results")
            return results
    except Exception as e:
        logger.debug(f"[Layer 2] Substack API failed: {e}")

    # 2. DDG fallback on substack.com
    try:
        ddg_results = duckduckgo_search.invoke(
            {"query": f"site:substack.com {query}", "max_results": max_results}
        )
        for r in ddg_results:
            r["source_type"] = "substack"
            results.append(r)
    except Exception as e:
        logger.warning(f"[Layer 2] Substack DDG fallback failed: {e}")

    logger.info(f"[Layer 2] Substack search: {len(results)} results")
    return results


@tool
def substack_fetch_post(url: str) -> dict:
    """
    Layer 2: Fetch a Substack post as clean text.

    Fetch chain:
      1. Substack post API (JSON — gives full content for public posts)
      2. Jina Reader (handles most Substack paywalls)
      3. Direct HTTP with session cookie (subscriber access)

    Returns:
        {url, markdown, title, authors, source_type}
    """
    logger.info(f"[Layer 2] Substack fetch: {url[:80]}")
    cfg = get_config()

    # 1. Try Substack JSON API — extract slug from URL
    try:
        # URL patterns: https://{sub}.substack.com/p/{slug}
        m = re.search(r"https?://([^.]+)\.substack\.com/p/([^/?#]+)", url)
        if m:
            subdomain, slug = m.group(1), m.group(2)
            api_url = f"https://{subdomain}.substack.com/api/v1/posts/{slug}"
            headers: dict[str, str] = {"Accept": "application/json"}
            if cfg.astra_substack_session_cookie:
                headers["Cookie"] = f"substack.sid={cfg.astra_substack_session_cookie}"
            resp = requests.get(api_url, timeout=15, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                body_html = data.get("body_html", "") or data.get("body", "")
                title = data.get("title", "")
                if body_html and len(body_html) > 300:
                    # Convert HTML to plain text
                    try:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(body_html, "html.parser")
                        text = soup.get_text("\n", strip=True)
                    except ImportError:
                        text = re.sub(r"<[^>]+>", " ", body_html)
                    authors_raw = data.get("publishedBylines", [{}])
                    authors = [a.get("name", "Unknown") for a in authors_raw]
                    logger.info(f"[Layer 2] Substack API: {len(text)} chars from {url[:60]}")
                    return {
                        "url": url, "markdown": text,
                        "title": title, "authors": authors, "source_type": "substack",
                    }
    except Exception as e:
        logger.debug(f"[Layer 2] Substack API fetch failed: {e}")

    # 2. Jina Reader
    try:
        jina_url = f"https://r.jina.ai/{url}"
        resp = httpx.get(
            jina_url, timeout=25, follow_redirects=True,
            headers={"Accept": "text/plain", "X-Return-Format": "markdown"},
        )
        resp.raise_for_status()
        text = resp.text.strip()
        if len(text) > 300:
            title = url
            for line in text.split("\n")[:5]:
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
            return {"url": url, "markdown": text, "title": title, "authors": [], "source_type": "substack"}
    except Exception as e:
        logger.debug(f"[Layer 2] Jina failed for Substack {url}: {e}")

    return {"url": url, "markdown": "", "title": url, "authors": [], "source_type": "substack"}


# ─── Papers With Code ─────────────────────────────────────────────────────────

@tool
def papers_with_code_search(
    query: str,
    max_results: int = 10,
) -> list[dict]:
    """
    Layer 2: Search Papers With Code for ML papers with implementations.

    Uses PWC public REST API (no auth required).

    Returns:
        List of {title, url, abstract, arxiv_id, github_url, tasks, source_type}.
    """
    logger.info(f"[Layer 2] Papers With Code: '{query[:60]}'")
    try:
        resp = requests.get(
            "https://paperswithcode.com/api/v1/papers/",
            params={"q": query, "items_per_page": max_results, "ordering": "-published"},
            timeout=15,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for paper in data.get("results", []):
            results.append({
                "title": paper.get("title", ""),
                "url": paper.get("url_abs", f"https://arxiv.org/abs/{paper.get('arxiv_id', '')}"),
                "snippet": paper.get("abstract", "")[:500],
                "arxiv_id": paper.get("arxiv_id", ""),
                "pdf_url": f"https://arxiv.org/pdf/{paper.get('arxiv_id', '')}" if paper.get("arxiv_id") else "",
                "github_url": paper.get("repository_url", ""),
                "tasks": [t.get("task", "") for t in paper.get("tasks", [])][:3],
                "source_type": "papers_with_code",
            })
        logger.info(f"[Layer 2] Papers With Code: {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"[Layer 2] Papers With Code failed: {e}")
        return []


# ─── Hugging Face Blog ────────────────────────────────────────────────────────

@tool
def huggingface_blog_search(query: str, max_results: int = 8) -> list[dict]:
    """
    Layer 2: Search Hugging Face Blog for technical articles.

    Uses DDG targeted search on huggingface.co/blog + hf.co/blog.

    Returns:
        List of {title, url, snippet, source_type} dicts.
    """
    logger.info(f"[Layer 2] HuggingFace Blog search: '{query[:60]}'")
    results: list[dict] = []
    try:
        ddg = duckduckgo_search.invoke({
            "query": f"site:huggingface.co/blog {query}",
            "max_results": max_results,
        })
        for r in ddg:
            r["source_type"] = "huggingface_blog"
            results.append(r)
    except Exception as e:
        logger.warning(f"[Layer 2] HF Blog search failed: {e}")

    # Also try the HF blog RSS for recent posts
    try:
        resp = requests.get(
            "https://huggingface.co/blog/feed.xml",
            timeout=10, headers={"Accept": "application/xml,text/xml"}
        )
        if resp.status_code == 200:
            from xml.etree import ElementTree as ET
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            query_lower = query.lower()
            for entry in root.findall(".//item") or root.findall(".//atom:entry", ns):
                title_el = entry.find("title") or entry.find("atom:title", ns)
                link_el = entry.find("link") or entry.find("atom:link", ns)
                desc_el = entry.find("description") or entry.find("atom:summary", ns)
                title_text = title_el.text if title_el is not None else ""
                link_text = (link_el.text or link_el.get("href", "")) if link_el is not None else ""
                desc_text = (desc_el.text or "")[:300] if desc_el is not None else ""
                if query_lower in (title_text + desc_text).lower():
                    results.append({
                        "title": title_text,
                        "url": link_text,
                        "snippet": re.sub(r"<[^>]+>", "", desc_text)[:300],
                        "source_type": "huggingface_blog",
                    })
                if len(results) >= max_results:
                    break
    except Exception:
        pass

    logger.info(f"[Layer 2] HuggingFace Blog: {len(results)} results")
    return results[:max_results]


# ─── The Gradient / Distill / Lilian Weng / Berkeley AI Blog ────────────────

@tool
def research_blogs_search(query: str, max_results: int = 10) -> list[dict]:
    """
    Layer 2: Search premier ML research blogs:
      - The Gradient (thegradient.pub)
      - Distill.pub (interactive ML papers)
      - Lilian Weng's blog (lilianweng.github.io)
      - Berkeley AI Research blog (bair.berkeley.edu/blog)
      - Google DeepMind blog (deepmind.google/blog)
      - Google AI blog (blog.research.google)
      - OpenAI blog (openai.com/research)
      - fast.ai (fast.ai)
      - Weights & Biases blog (wandb.ai/blog)

    Returns:
        List of {title, url, snippet, source_type} dicts.
    """
    RESEARCH_BLOG_SITES = [
        "site:thegradient.pub",
        "site:distill.pub",
        "site:lilianweng.github.io",
        "site:bair.berkeley.edu/blog",
        "site:deepmind.google/blog",
        "site:blog.research.google",
        "site:openai.com/research",
        "site:fast.ai",
        "site:wandb.ai/blog",
        "site:neptune.ai/blog",
    ]
    site_filter = " OR ".join(RESEARCH_BLOG_SITES[:5])  # DDG handles ~5 site: ops
    full_query = f"({site_filter}) {query}"

    logger.info(f"[Layer 2] Research blogs search: '{query[:50]}'")
    results: list[dict] = []
    try:
        ddg = duckduckgo_search.invoke({"query": full_query, "max_results": max_results})
        for r in ddg:
            r["source_type"] = "research_blog"
            results.append(r)
    except Exception as e:
        logger.warning(f"[Layer 2] Research blogs DDG failed: {e}")

    logger.info(f"[Layer 2] Research blogs: {len(results)} results")
    return results


# ─── Aggregator (used in code-action blocks) ─────────────────────────────────

@tool
def wikipedia_search(
    query: str,
    max_results: int = 5,
    lang: str = "en",
) -> list[dict]:
    """
    Layer 2: Search Wikipedia and return article summaries with full intro text.
    Useful for background/overview material on well-established topics.

    Args:
        query: Search query string.
        max_results: Maximum number of Wikipedia articles to return (default 5).
        lang: Wikipedia language code (default "en").

    Returns:
        List of {url, title, snippet, raw_content, source_type} dicts.
    """
    logger.info(f"[Layer 2] Wikipedia: '{query[:60]}' max={max_results}")

    try:
        import wikipedia as wiki_lib  # pip install wikipedia>=1.4.0

        wiki_lib.set_lang(lang)
        try:
            titles = wiki_lib.search(query, results=max_results)
        except Exception as e:
            logger.warning(f"[Layer 2] Wikipedia search failed: {e}")
            return []

        results: list[dict] = []
        for title in titles[:max_results]:
            try:
                page = wiki_lib.page(title, auto_suggest=False)
                results.append({
                    "url": page.url,
                    "title": page.title,
                    "snippet": page.summary[:500],
                    "raw_content": page.content[:8000],
                    "source_type": "wikipedia",
                })
            except wiki_lib.exceptions.DisambiguationError:
                continue
            except wiki_lib.exceptions.PageError:
                continue
            except Exception as e:
                logger.debug(f"[Layer 2] Wikipedia page error '{title}': {e}")
                continue

        logger.info(f"[Layer 2] Wikipedia returned {len(results)} articles")
        return results

    except ImportError:
        logger.warning("[Layer 2] wikipedia package not installed — skipping Wikipedia search")
        return []
    except Exception as e:
        logger.error(f"[Layer 2] Wikipedia search error: {e}")
        return []


def deduplicate_sources(all_results: list[dict]) -> list[dict]:
    """
    Deduplicate collected sources across all crawlers by URL.
    Used in code-action aggregation steps.
    """
    seen_urls: set[str] = set()
    unique: list[dict] = []
    for r in all_results:
        url = r.get("url") or r.get("pdf_url") or ""
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique.append(r)
        elif not url:
            unique.append(r)  # Keep items without URLs
    logger.info(f"[Layer 2] Deduplicated: {len(all_results)} → {len(unique)} sources")
    return unique
