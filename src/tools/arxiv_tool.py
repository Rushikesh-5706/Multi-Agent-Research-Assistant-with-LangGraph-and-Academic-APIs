from __future__ import annotations

from typing import List

import arxiv
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.models.schemas import Paper

_MAX_RESULTS = 5


class ArxivTool:
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def search(self, query: str) -> List[Paper]:
        logger.info(f"Search Agent: Querying arXiv | query={query!r}")
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=_MAX_RESULTS,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers: List[Paper] = []
        for result in client.results(search):
            arxiv_id = result.entry_id.split("/")[-1]
            paper = Paper(
                paper_id=f"arxiv:{arxiv_id}",
                title=result.title.strip(),
                authors=[a.name for a in result.authors],
                abstract=result.summary.strip(),
                year=result.published.year if result.published else None,
                pdf_url=result.pdf_url,
                arxiv_id=arxiv_id,
                source="arxiv",
            )
            papers.append(paper)
        logger.info(
            f"Search Agent: Found {len(papers)} papers from arXiv | query={query!r}"
        )
        return papers
