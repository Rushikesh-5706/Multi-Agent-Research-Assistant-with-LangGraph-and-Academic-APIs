from unittest.mock import MagicMock, patch

import pytest

from src.agents.search_agent import SearchAgent
from src.models.schemas import Paper, WorkflowState


def _make_paper(title: str, source: str) -> Paper:
    return Paper(
        paper_id=f"{source}:001",
        title=title,
        authors=["Author One"],
        abstract="Abstract.",
        source=source,
    )


@pytest.fixture
def mock_search_agent():
    with patch("src.agents.search_agent.ArxivTool") as mock_arxiv_cls, \
         patch("src.agents.search_agent.SemanticScholarTool") as mock_ss_cls:
        mock_arxiv = MagicMock()
        mock_ss = MagicMock()
        mock_arxiv_cls.return_value = mock_arxiv
        mock_ss_cls.return_value = mock_ss
        agent = SearchAgent()
        yield agent, mock_arxiv, mock_ss


def test_search_deduplicates_by_title(mock_search_agent):
    agent, mock_arxiv, mock_ss = mock_search_agent
    duplicate = _make_paper("Duplicate Title", "arxiv")
    mock_arxiv.search.return_value = [duplicate]
    mock_ss.search.return_value = [_make_paper("Duplicate Title", "semantic_scholar")]

    ws = WorkflowState(run_id="r1", topic="nlp", search_queries=["nlp"])
    result = agent.search(ws)

    assert len(result.papers) == 1
    assert result.stage == "searched"


def test_search_caps_at_max_papers(mock_search_agent):
    agent, mock_arxiv, mock_ss = mock_search_agent
    arxiv_papers = [_make_paper(f"Paper {i}", "arxiv") for i in range(8)]
    ss_papers = [_make_paper(f"SS Paper {i}", "semantic_scholar") for i in range(8)]
    mock_arxiv.search.return_value = arxiv_papers
    mock_ss.search.return_value = ss_papers

    ws = WorkflowState(run_id="r1", topic="nlp", search_queries=["nlp", "survey nlp"])
    result = agent.search(ws)

    assert len(result.papers) <= 10


def test_search_continues_if_arxiv_fails(mock_search_agent):
    agent, mock_arxiv, mock_ss = mock_search_agent
    mock_arxiv.search.side_effect = Exception("arXiv down")
    mock_ss.search.return_value = [_make_paper("SS Paper 1", "semantic_scholar")]

    ws = WorkflowState(run_id="r1", topic="nlp", search_queries=["nlp"])
    result = agent.search(ws)

    assert len(result.papers) == 1
    assert result.papers[0].source == "semantic_scholar"


def test_search_continues_if_semantic_scholar_fails(mock_search_agent):
    agent, mock_arxiv, mock_ss = mock_search_agent
    mock_arxiv.search.return_value = [_make_paper("arXiv Paper 1", "arxiv")]
    mock_ss.search.side_effect = Exception("SS down")

    ws = WorkflowState(run_id="r1", topic="nlp", search_queries=["nlp"])
    result = agent.search(ws)

    assert len(result.papers) == 1
    assert result.papers[0].source == "arxiv"


def test_search_sets_stage(mock_search_agent):
    agent, mock_arxiv, mock_ss = mock_search_agent
    mock_arxiv.search.return_value = [_make_paper("Paper A", "arxiv")]
    mock_ss.search.return_value = []

    ws = WorkflowState(run_id="r1", topic="nlp", search_queries=["nlp"])
    result = agent.search(ws)
    assert result.stage == "searched"


def test_search_uses_all_queries(mock_search_agent):
    agent, mock_arxiv, mock_ss = mock_search_agent
    mock_arxiv.search.return_value = []
    mock_ss.search.return_value = []

    ws = WorkflowState(run_id="r1", topic="nlp", search_queries=["nlp", "survey nlp"])
    agent.search(ws)

    assert mock_arxiv.search.call_count == 2
    assert mock_ss.search.call_count == 2
