from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Paper(BaseModel):
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    year: Optional[int] = None
    pdf_url: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    source: str  # "arxiv" or "semantic_scholar"
    summary: Optional[str] = None
    bibtex: Optional[str] = None
    bibtex_key: Optional[str] = None


class WorkflowState(BaseModel):
    run_id: str
    topic: str
    search_queries: List[str] = Field(default_factory=list)
    papers: List[Paper] = Field(default_factory=list)
    stage: str = "init"
    error: Optional[str] = None
