from __future__ import annotations

import os
from typing import List

import requests
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.models.schemas import Paper

_SS_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "title,authors,abstract,year,openAccessPdf,externalIds"
_MAX_RESULTS = 5


class SemanticScholarTool:
    def __init__(self) -> None:
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        self._headers = {"x-api-key": api_key} if api_key else {}

    @retry(
        wait=wait_exponential(multiplier=1, min=3, max=12),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True,
    )
    def search(self, query: str) -> List[Paper]:
        logger.info(f"Search Agent: Querying Semantic Scholar | query={query!r}")
        params = {"query": query, "limit": _MAX_RESULTS, "fields": _FIELDS}
        response = requests.get(
            f"{_SS_BASE}/paper/search",
            params=params,
            headers=self._headers,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        papers: List[Paper] = []
        for item in data.get("data", []):
            abstract = (item.get("abstract") or "").strip()
            if not abstract:
                continue
            external_ids = item.get("externalIds") or {}
            doi = external_ids.get("DOI")
            arxiv_id = external_ids.get("ArXiv")
            pdf_url: str | None = None
            open_access = item.get("openAccessPdf")
            if open_access:
                pdf_url = open_access.get("url")
            paper = Paper(
                paper_id=f"ss:{item['paperId']}",
                title=(item.get("title") or "Untitled").strip(),
                authors=[a.get("name", "") for a in (item.get("authors") or [])],
                abstract=abstract,
                year=item.get("year"),
                pdf_url=pdf_url,
                doi=doi,
                arxiv_id=arxiv_id,
                source="semantic_scholar",
            )
            papers.append(paper)
        logger.info(
            f"Search Agent: Found {len(papers)} papers from Semantic Scholar | query={query!r}"
        )
        return papers
