from __future__ import annotations

from typing import List

from loguru import logger

from src.models.schemas import Paper, WorkflowState
from src.tools.arxiv_tool import ArxivTool
from src.tools.semantic_scholar_tool import SemanticScholarTool

_MAX_PAPERS = 10


class SearchAgent:
    def __init__(self) -> None:
        self._arxiv = ArxivTool()
        self._ss = SemanticScholarTool()

    def search(self, state: WorkflowState) -> WorkflowState:
        logger.info(
            f"Search Agent: Starting retrieval | queries={len(state.search_queries)}"
        )
        seen_titles: set[str] = set()
        papers: List[Paper] = []

        for query in state.search_queries:
            try:
                arxiv_papers = self._arxiv.search(query)
                for p in arxiv_papers:
                    key = p.title.lower().strip()
                    if key not in seen_titles:
                        seen_titles.add(key)
                        papers.append(p)
            except Exception as exc:
                logger.error(
                    f"Search Agent: arXiv failed after retries | query={query!r} | error={exc}"
                )

            try:
                ss_papers = self._ss.search(query)
                for p in ss_papers:
                    key = p.title.lower().strip()
                    if key not in seen_titles:
                        seen_titles.add(key)
                        papers.append(p)
            except Exception as exc:
                logger.error(
                    f"Search Agent: Semantic Scholar failed after retries | query={query!r} | error={exc}"
                )

        papers = papers[:_MAX_PAPERS]
        state.papers = papers
        state.stage = "searched"

        sources = {p.source for p in papers}
        logger.info(
            f"Search Agent: Retrieved {len(papers)} unique papers | sources={sources}"
        )
        return state
